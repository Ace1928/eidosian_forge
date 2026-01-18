import collections
import hashlib
import os
import random
import threading
import warnings
from datetime import datetime
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import objective as obj_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
def _duplicate(self, values):
    """Check if the values has been tried in previous trials.

        Args:
            A dictionary mapping hyperparameter names to suggested values.

        Returns:
            Boolean. Whether the values has been tried in previous trials.
        """
    return self._compute_values_hash(values) in self._tried_so_far