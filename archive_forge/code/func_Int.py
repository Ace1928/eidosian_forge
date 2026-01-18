import collections
import contextlib
import copy
import six
from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import conditions as conditions_mod
from keras_tuner.src.engine.hyperparameters import hp_types
from keras_tuner.src.engine.hyperparameters import hyperparameter as hp_module
def Int(self, name, min_value, max_value, step=None, sampling='linear', default=None, parent_name=None, parent_values=None):
    """Integer hyperparameter.

        Note that unlike Python's `range` function, `max_value` is *included* in
        the possible values this parameter can take on.


        Example #1:

        ```py
        hp.Int(
            "n_layers",
            min_value=6,
            max_value=12)
        ```

        The possible values are [6, 7, 8, 9, 10, 11, 12].

        Example #2:

        ```py
        hp.Int(
            "n_layers",
            min_value=6,
            max_value=13,
            step=3)
        ```

        `step` is the minimum distance between samples.
        The possible values are [6, 9, 12].

        Example #3:

        ```py
        hp.Int(
            "batch_size",
            min_value=2,
            max_value=32,
            step=2,
            sampling="log")
        ```

        When `sampling="log"` the `step` is multiplied between samples.
        The possible values are [2, 4, 8, 16, 32].

        Args:
            name: A string. the name of parameter. Must be unique for each
                `HyperParameter` instance in the search space.
            min_value: Integer, the lower limit of range, inclusive.
            max_value: Integer, the upper limit of range, inclusive.
            step: Optional integer, the distance between two consecutive samples
                in the range. If left unspecified, it is possible to sample any
                integers in the interval. If `sampling="linear"`, it will be the
                minimum additve between two samples. If `sampling="log"`, it
                will be the minimum multiplier between two samples.
            sampling: String. One of "linear", "log", "reverse_log". Defaults to
                "linear". When sampling value, it always start from a value in
                range [0.0, 1.0). The `sampling` argument decides how the value
                is projected into the range of [min_value, max_value].
                "linear": min_value + value * (max_value - min_value)
                "log": min_value * (max_value / min_value) ^ value
                "reverse_log":
                    (max_value -
                     min_value * ((max_value / min_value) ^ (1 - value) - 1))
            default: Integer, default value to return for the parameter. If
                unspecified, the default value will be `min_value`.
            parent_name: Optional string, specifying the name of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.
            parent_values: Optional list of the values of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.

        Returns:
            The value of the hyperparameter, or None if the hyperparameter is
            not active.
        """
    with self._maybe_conditional_scope(parent_name, parent_values):
        hp = hp_types.Int(name=self._get_name(name), min_value=min_value, max_value=max_value, step=step, sampling=sampling, default=default, conditions=self._conditions)
        return self._retrieve(hp)