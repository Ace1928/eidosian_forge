import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
import torch
def _crop_past_key_values(model, past_key_values, maximum_length):
    """Crops the past key values up to a certain maximum length."""
    new_past = []
    if model.config.is_encoder_decoder:
        for idx in range(len(past_key_values)):
            new_past.append((past_key_values[idx][0][:, :, :maximum_length, :], past_key_values[idx][1][:, :, :maximum_length, :], past_key_values[idx][2], past_key_values[idx][3]))
        past_key_values = tuple(new_past)
    elif 'bloom' in model.__class__.__name__.lower() or (model.config.architectures is not None and 'bloom' in model.config.architectures[0].lower()):
        for idx in range(len(past_key_values)):
            new_past.append((past_key_values[idx][0][:, :, :maximum_length], past_key_values[idx][1][:, :maximum_length, :]))
        past_key_values = tuple(new_past)
    elif 'gptbigcode' in model.__class__.__name__.lower() or (model.config.architectures is not None and 'gptbigcode' in model.config.architectures[0].lower()):
        if model.config.multi_query:
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :maximum_length, :]
        else:
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :, :maximum_length, :]
    else:
        for idx in range(len(past_key_values)):
            new_past.append((past_key_values[idx][0][:, :, :maximum_length, :], past_key_values[idx][1][:, :, :maximum_length, :]))
        past_key_values = tuple(new_past)
    return past_key_values