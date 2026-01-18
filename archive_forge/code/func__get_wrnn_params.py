import logging
import os
import torch
from torchaudio._internal import download_url_to_file, module_utils as _mod_utils
def _get_wrnn_params():
    return {'upsample_scales': [5, 5, 11], 'n_classes': 2 ** 8, 'hop_length': 275, 'n_res_block': 10, 'n_rnn': 512, 'n_fc': 512, 'kernel_size': 5, 'n_freq': 80, 'n_hidden': 128, 'n_output': 128}