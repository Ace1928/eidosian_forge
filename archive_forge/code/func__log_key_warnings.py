import os
import re
import numpy
from .utils import (
from .utils import transpose as transpose_func
def _log_key_warnings(missing_keys, unexpected_keys, mismatched_keys, class_name):
    if len(unexpected_keys) > 0:
        logger.warning(f'Some weights of the PyTorch model were not used when initializing the TF 2.0 model {class_name}: {unexpected_keys}\n- This IS expected if you are initializing {class_name} from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).\n- This IS NOT expected if you are initializing {class_name} from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).')
    else:
        logger.warning(f'All PyTorch model weights were used when initializing {class_name}.\n')
    if len(missing_keys) > 0:
        logger.warning(f'Some weights or buffers of the TF 2.0 model {class_name} were not initialized from the PyTorch model and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')
    else:
        logger.warning(f'All the weights of {class_name} were initialized from the PyTorch model.\nIf your task is similar to the task the model of the checkpoint was trained on, you can already use {class_name} for predictions without further training.')
    if len(mismatched_keys) > 0:
        mismatched_warning = '\n'.join([f'- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated' for key, shape1, shape2 in mismatched_keys])
        logger.warning(f'Some weights of {class_name} were not initialized from the model checkpoint are newly initialized because the shapes did not match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.')