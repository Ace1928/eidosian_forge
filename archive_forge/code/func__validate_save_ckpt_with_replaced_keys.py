from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import os
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _validate_save_ckpt_with_replaced_keys(new_copy, replaced_keys):
    """Validates the save ckpt properties."""
    save_steps = new_copy.save_checkpoints_steps
    save_secs = new_copy.save_checkpoints_secs
    if 'save_checkpoints_steps' in replaced_keys and 'save_checkpoints_secs' in replaced_keys:
        if save_steps is not None and save_secs is not None:
            raise ValueError(_SAVE_CKPT_ERR)
    elif 'save_checkpoints_steps' in replaced_keys and save_steps is not None:
        new_copy._save_checkpoints_secs = None
    elif 'save_checkpoints_secs' in replaced_keys and save_secs is not None:
        new_copy._save_checkpoints_steps = None