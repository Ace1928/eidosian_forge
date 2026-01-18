import copy
import os
import traceback
import warnings
from keras_tuner.src import backend
from keras_tuner.src import config as config_module
from keras_tuner.src import errors
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.distribute import utils as dist_utils
from keras_tuner.src.engine import hypermodel as hm_module
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
from keras_tuner.src.engine import tuner_utils
def _populate_initial_space(self):
    """Populate initial search space for oracle.

        Keep this function as a subroutine for AutoKeras to override. The space
        may not be ready at the initialization of the tuner, but after seeing
        the training data.

        Build hypermodel multiple times to find all conditional hps. It
        generates hp values based on the not activated `conditional_scopes`
        found in the builds.
        """
    if self.hypermodel is None:
        return
    hp = self.oracle.get_space()
    self.hypermodel.declare_hyperparameters(hp)
    self.oracle.update_space(hp)
    self._activate_all_conditions()