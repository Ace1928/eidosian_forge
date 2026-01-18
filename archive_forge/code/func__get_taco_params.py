import logging
import os
import torch
from torchaudio._internal import download_url_to_file, module_utils as _mod_utils
def _get_taco_params(n_symbols):
    return {'mask_padding': False, 'n_mels': 80, 'n_frames_per_step': 1, 'symbol_embedding_dim': 512, 'encoder_embedding_dim': 512, 'encoder_n_convolution': 3, 'encoder_kernel_size': 5, 'decoder_rnn_dim': 1024, 'decoder_max_step': 2000, 'decoder_dropout': 0.1, 'decoder_early_stopping': True, 'attention_rnn_dim': 1024, 'attention_hidden_dim': 128, 'attention_location_n_filter': 32, 'attention_location_kernel_size': 31, 'attention_dropout': 0.1, 'prenet_dim': 256, 'postnet_n_convolution': 5, 'postnet_kernel_size': 5, 'postnet_embedding_dim': 512, 'gate_threshold': 0.5, 'n_symbol': n_symbols}