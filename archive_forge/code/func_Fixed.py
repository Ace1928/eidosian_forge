import collections
import contextlib
import copy
import six
from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import conditions as conditions_mod
from keras_tuner.src.engine.hyperparameters import hp_types
from keras_tuner.src.engine.hyperparameters import hyperparameter as hp_module
def Fixed(self, name, value, parent_name=None, parent_values=None):
    """Fixed, untunable value.

        Args:
            name: A string. the name of parameter. Must be unique for each
                `HyperParameter` instance in the search space.
            value: The value to use (can be any JSON-serializable Python type).
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
        hp = hp_types.Fixed(name=self._get_name(name), value=value, conditions=self._conditions)
        return self._retrieve(hp)