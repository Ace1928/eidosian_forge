import argparse
import os
from pathlib import Path
import torch
from accelerate.utils.modeling import find_tied_parameters
from seamless_communication.models.inference.translator import Translator
from transformers import (
from transformers.utils import logging
def _convert_model(original_model, hf_model, convert_list, device, unwanted_prefix='model.', filter_state_dict='speech', exclude_state_dict=None):
    state_dict = original_model.state_dict()
    if isinstance(filter_state_dict, str):

        def filter_func(x):
            return filter_state_dict in x[0]
    else:

        def filter_func(item):
            if exclude_state_dict is not None and exclude_state_dict in item[0]:
                return False
            for filter_el in filter_state_dict:
                if filter_el in item[0]:
                    return True
            return False
    state_dict = dict(filter(filter_func, state_dict.items()))
    for k, v in list(state_dict.items()):
        new_k = k[len(unwanted_prefix):]
        for old_layer_name, new_layer_name in convert_list:
            if old_layer_name in new_k:
                new_k = new_k.replace(old_layer_name, new_layer_name)
        if '.layer_norm' in new_k and new_k.split('.layer_norm')[0][-1].isnumeric():
            new_k = new_k.replace('layer_norm', 'final_layer_norm')
        state_dict[new_k] = state_dict.pop(k)
    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    extra_keys = set(extra_keys)
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set({k for k in missing_keys if 'final_logits_bias' not in k})
    if len(extra_keys) != 0:
        raise ValueError(f'extra keys found: {extra_keys}')
    if len(missing_keys) != 0:
        raise ValueError(f'missing keys: {missing_keys}')
    hf_model.load_state_dict(state_dict, strict=False)
    n_params = param_count(hf_model)
    logger.info(f'model loaded: {round(n_params / 1000000.0, 1)}M params')
    hf_model.eval()
    hf_model.to(device)
    del state_dict
    return hf_model