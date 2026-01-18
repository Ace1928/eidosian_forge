import collections
import copy
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner as tuner_module
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