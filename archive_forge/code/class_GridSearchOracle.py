import collections
import copy
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner as tuner_module
@keras_tuner_export('keras_tuner.oracles.GridSearchOracle')
class GridSearchOracle(oracle_module.Oracle):
    """Grid search oracle.

    Args:
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        max_trials: Optional integer, the total number of trials (model
            configurations) to test at most. Note that the oracle may interrupt
            the search before `max_trial` models have been tested if the search
            space has been exhausted. If left unspecified, it runs till the
            search space is exhausted.
        seed: Optional integer, the random seed.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
        tune_new_entries: Boolean, whether hyperparameter entries that are
            requested by the hypermodel but that were not specified in
            `hyperparameters` should be added to the search space, or not. If
            not, then the default value for these parameters will be used.
            Defaults to True.
        allow_new_entries: Boolean, whether the hypermodel is allowed to
            request hyperparameter entries not listed in `hyperparameters`.
            Defaults to True.
        max_retries_per_trial: Integer. Defaults to 0. The maximum number of
            times to retry a `Trial` if the trial crashed or the results are
            invalid.
        max_consecutive_failed_trials: Integer. Defaults to 3. The maximum
            number of consecutive failed `Trial`s. When this number is reached,
            the search will be stopped. A `Trial` is marked as failed when none
            of the retries succeeded.
    """

    def __init__(self, objective=None, max_trials=None, seed=None, hyperparameters=None, allow_new_entries=True, tune_new_entries=True, max_retries_per_trial=0, max_consecutive_failed_trials=3):
        super().__init__(objective=objective, max_trials=max_trials, hyperparameters=hyperparameters, tune_new_entries=tune_new_entries, allow_new_entries=allow_new_entries, seed=seed, max_retries_per_trial=max_retries_per_trial, max_consecutive_failed_trials=max_consecutive_failed_trials)
        self._ordered_ids = LinkedList()
        self._populate_next = []

    def populate_space(self, trial_id):
        """Fill the hyperparameter space with values.

        Args:
            trial_id: A string, the ID for this Trial.

        Returns:
            A dictionary with keys "values" and "status", where "values" is
            a mapping of parameter names to suggested values, and "status"
            should be one of "RUNNING" (the trial can start normally), "IDLE"
            (the oracle is waiting on something and cannot create a trial), or
            "STOPPED" (the oracle has finished searching and no new trial should
            be created).
        """
        values = None
        if len(self.start_order) == 0:
            self._ordered_ids.insert(trial_id)
            hps = self.get_space()
            values = {hp.name: hp.default for hp in self.get_space().space if hps.is_active(hp)}
            self._populate_next.append(trial_id)
        while len(self._populate_next) > 0 and values is None:
            old_trial_id = self._populate_next.pop(0)
            old_values = self.trials[old_trial_id].hyperparameters.values
            new_values = self._get_next_combination(old_values)
            if new_values is None:
                continue
            next_id = self._ordered_ids.next(old_trial_id)
            if next_id is not None:
                next_values = self.trials[next_id].hyperparameters.values
                if self._compare(new_values, next_values) >= 0:
                    continue
            self._ordered_ids.insert(trial_id, old_trial_id)
            values = new_values
        if values is not None:
            return {'status': trial_module.TrialStatus.RUNNING, 'values': values}
        if len(self.ongoing_trials) > 0:
            return {'status': trial_module.TrialStatus.IDLE, 'values': None}
        return {'status': trial_module.TrialStatus.STOPPED, 'values': None}

    def _compare(self, a, b):
        """Compare two `HyperParameters`' values.

        The smallest index where a differs from b decides which one is larger.
        In the values of one `HyperParameter`, the default value is the
        smallest. The rest are sorted according to their order in
        `HyperParameter.values`.  If one value is the prefix of another, the
        longer one is larger.

        Args:
            a: Dict. HyperParameters values. Only active values are included.
            b: Dict. HyperParameters values. Only active values are included.

        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b.
        """
        hps = self.get_space()
        for hp in hps.space:
            if hp.name not in a:
                continue
            if a[hp.name] == b[hp.name]:
                continue
            value_list = list(hp.values)
            if hp.default in value_list:
                value_list.remove(hp.default)
            value_list.insert(0, hp.default)
            index_a = value_list.index(a[hp.name])
            index_b = value_list.index(b[hp.name])
            return -1 if index_a < index_b else 1
        return 0

    def _get_next_combination(self, values):
        """Get the next value combination to try.

        Given the last trial's values dictionary, this method retrieves the next
        hyperparameter values to try. As it requires the last trial's
        values as input, it should not be called on the first trial. The first
        trial will always use default hp values.

        This oracle iterates over the search space entirely deterministically.

        When a new hp appears in a trial, the first value tried for that hp
        will be its default value.

        Args:
            values: Dict. The keys are hp names. The values are the hp values
                from the last trial.

        Returns:
            Dict or None. The next possible value combination for the
            hyperparameters. If no next combination exist (values is the last
            combination), it returns None. The return values only include the
            active ones.
        """
        hps = self.get_space()
        all_values = {}
        for hp in hps.space:
            value_list = list(hp.values)
            if hp.default in value_list:
                value_list.remove(hp.default)
            all_values[hp.name] = [hp.default] + value_list
        default_values = {hp.name: hp.default for hp in hps.space}
        hps.values = copy.deepcopy(values)
        bumped_value = False
        for hp in reversed(hps.space):
            name = hp.name
            if hps.is_active(hp):
                value = hps.values[name]
                if value != all_values[name][-1]:
                    index = all_values[name].index(value) + 1
                    hps.values[name] = all_values[name][index]
                    bumped_value = True
                    break
            hps.values[name] = default_values[name]
        hps.ensure_active_values()
        return hps.values if bumped_value else None

    @oracle_module.synchronized
    def end_trial(self, trial):
        super().end_trial(trial)
        self._populate_next.append(trial.trial_id)