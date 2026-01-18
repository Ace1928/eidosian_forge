import copy
import math
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import tuner as tuner_module
@keras_tuner_export(['keras_tuner.Hyperband', 'keras_tuner.tuners.Hyperband'])
class Hyperband(tuner_module.Tuner):
    """Variation of HyperBand algorithm.

    Reference:
        Li, Lisha, and Kevin Jamieson.
        ["Hyperband: A Novel Bandit-Based
         Approach to Hyperparameter Optimization."
        Journal of Machine Learning Research 18 (2018): 1-52](
            http://jmlr.org/papers/v18/16-558.html).


    Args:
        hypermodel: Instance of `HyperModel` class (or callable that takes
            hyperparameters and returns a `Model` instance). It is optional
            when `Tuner.run_trial()` is overriden and does not use
            `self.hypermodel`.
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        max_epochs: Integer, the maximum number of epochs to train one model.
            It is recommended to set this to a value slightly higher than the
            expected epochs to convergence for your largest Model, and to use
            early stopping during training (for example, via
            `tf.keras.callbacks.EarlyStopping`). Defaults to 100.
        factor: Integer, the reduction factor for the number of epochs
            and number of models for each bracket. Defaults to 3.
        hyperband_iterations: Integer, at least 1, the number of times to
            iterate over the full Hyperband algorithm. One iteration will run
            approximately `max_epochs * (math.log(max_epochs, factor) ** 2)`
            cumulative epochs across all trials. It is recommended to set this
            to as high a value as is within your resource budget. Defaults to
            1.
        seed: Optional integer, the random seed.
        hyperparameters: Optional HyperParameters instance. Can be used to
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
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """

    def __init__(self, hypermodel=None, objective=None, max_epochs=100, factor=3, hyperband_iterations=1, seed=None, hyperparameters=None, tune_new_entries=True, allow_new_entries=True, max_retries_per_trial=0, max_consecutive_failed_trials=3, **kwargs):
        oracle = HyperbandOracle(objective, max_epochs=max_epochs, factor=factor, hyperband_iterations=hyperband_iterations, seed=seed, hyperparameters=hyperparameters, tune_new_entries=tune_new_entries, allow_new_entries=allow_new_entries, max_retries_per_trial=max_retries_per_trial, max_consecutive_failed_trials=max_consecutive_failed_trials)
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        hp = trial.hyperparameters
        if 'tuner/epochs' in hp.values:
            fit_kwargs['epochs'] = hp.values['tuner/epochs']
            fit_kwargs['initial_epoch'] = hp.values['tuner/initial_epoch']
        return super().run_trial(trial, *fit_args, **fit_kwargs)

    def _build_hypermodel(self, hp):
        model = super()._build_hypermodel(hp)
        if 'tuner/trial_id' in hp.values:
            trial_id = hp.values['tuner/trial_id']
            if backend.config.multi_backend():
                model.build_from_config(utils.load_json(self._get_build_config_fname(trial_id)))
            model.load_weights(self._get_checkpoint_fname(trial_id))
        return model