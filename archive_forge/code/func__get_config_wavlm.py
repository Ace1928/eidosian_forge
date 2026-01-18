import logging
from typing import Any, Dict
import torch
from torch.nn import Module
from ..model import wav2vec2_model, Wav2Vec2Model, wavlm_model
def _get_config_wavlm(cfg):
    config = {'extractor_mode': f'{cfg.feat_extract_norm}_norm', 'extractor_conv_layer_config': list(zip(cfg.conv_dim, cfg.conv_kernel, cfg.conv_stride)), 'extractor_conv_bias': cfg.conv_bias, 'encoder_embed_dim': cfg.hidden_size, 'encoder_projection_dropout': cfg.feat_proj_dropout, 'encoder_pos_conv_kernel': cfg.num_conv_pos_embeddings, 'encoder_pos_conv_groups': cfg.num_conv_pos_embedding_groups, 'encoder_num_layers': cfg.num_hidden_layers, 'encoder_num_heads': cfg.num_attention_heads, 'encoder_num_buckets': cfg.num_buckets, 'encoder_max_distance': cfg.max_bucket_distance, 'encoder_attention_dropout': cfg.attention_dropout, 'encoder_ff_interm_features': cfg.intermediate_size, 'encoder_ff_interm_dropout': cfg.activation_dropout, 'encoder_dropout': cfg.hidden_dropout, 'encoder_layer_norm_first': cfg.do_stable_layer_norm, 'encoder_layer_drop': cfg.layerdrop}
    return config