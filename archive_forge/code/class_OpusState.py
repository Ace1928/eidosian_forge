import argparse
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile
import numpy as np
import torch
from huggingface_hub.hf_api import list_models
from torch import nn
from tqdm import tqdm
from transformers import MarianConfig, MarianMTModel, MarianTokenizer
class OpusState:

    def __init__(self, source_dir, eos_token_id=0):
        npz_path = find_model_file(source_dir)
        self.state_dict = np.load(npz_path)
        cfg = load_config_from_state_dict(self.state_dict)
        if cfg['dim-vocabs'][0] != cfg['dim-vocabs'][1]:
            raise ValueError
        if 'Wpos' in self.state_dict:
            raise ValueError('Wpos key in state dictionary')
        self.state_dict = dict(self.state_dict)
        if cfg['tied-embeddings-all']:
            cfg['tied-embeddings-src'] = True
            cfg['tied-embeddings'] = True
        self.share_encoder_decoder_embeddings = cfg['tied-embeddings-src']
        self.source_dir = source_dir
        self.tokenizer = self.load_tokenizer()
        tokenizer_has_eos_token_id = hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None
        eos_token_id = self.tokenizer.eos_token_id if tokenizer_has_eos_token_id else 0
        if cfg['tied-embeddings-src']:
            self.wemb, self.final_bias = add_emb_entries(self.state_dict['Wemb'], self.state_dict[BIAS_KEY], 1)
            self.pad_token_id = self.wemb.shape[0] - 1
            cfg['vocab_size'] = self.pad_token_id + 1
        else:
            self.wemb, _ = add_emb_entries(self.state_dict['encoder_Wemb'], self.state_dict[BIAS_KEY], 1)
            self.dec_wemb, self.final_bias = add_emb_entries(self.state_dict['decoder_Wemb'], self.state_dict[BIAS_KEY], 1)
            self.pad_token_id = self.wemb.shape[0] - 1
            cfg['vocab_size'] = self.pad_token_id + 1
            cfg['decoder_vocab_size'] = self.pad_token_id + 1
        if cfg['vocab_size'] != self.tokenizer.vocab_size:
            raise ValueError(f'Original vocab size {cfg['vocab_size']} and new vocab size {len(self.tokenizer.encoder)} mismatched.')
        self.state_keys = list(self.state_dict.keys())
        if 'Wtype' in self.state_dict:
            raise ValueError('Wtype key in state dictionary')
        self._check_layer_entries()
        self.cfg = cfg
        hidden_size, intermediate_shape = self.state_dict['encoder_l1_ffn_W1'].shape
        if hidden_size != cfg['dim-emb']:
            raise ValueError(f'Hidden size {hidden_size} and configured size {cfg['dim_emb']} mismatched')
        decoder_yml = cast_marian_config(load_yaml(source_dir / 'decoder.yml'))
        check_marian_cfg_assumptions(cfg)
        self.hf_config = MarianConfig(vocab_size=cfg['vocab_size'], decoder_vocab_size=cfg.get('decoder_vocab_size', cfg['vocab_size']), share_encoder_decoder_embeddings=cfg['tied-embeddings-src'], decoder_layers=cfg['dec-depth'], encoder_layers=cfg['enc-depth'], decoder_attention_heads=cfg['transformer-heads'], encoder_attention_heads=cfg['transformer-heads'], decoder_ffn_dim=cfg['transformer-dim-ffn'], encoder_ffn_dim=cfg['transformer-dim-ffn'], d_model=cfg['dim-emb'], activation_function=cfg['transformer-ffn-activation'], pad_token_id=self.pad_token_id, eos_token_id=eos_token_id, forced_eos_token_id=eos_token_id, bos_token_id=0, max_position_embeddings=cfg['dim-emb'], scale_embedding=True, normalize_embedding='n' in cfg['transformer-preprocess'], static_position_embeddings=not cfg['transformer-train-position-embeddings'], tie_word_embeddings=cfg['tied-embeddings'], dropout=0.1, num_beams=decoder_yml['beam-size'], decoder_start_token_id=self.pad_token_id, bad_words_ids=[[self.pad_token_id]], max_length=512)

    def _check_layer_entries(self):
        self.encoder_l1 = self.sub_keys('encoder_l1')
        self.decoder_l1 = self.sub_keys('decoder_l1')
        self.decoder_l2 = self.sub_keys('decoder_l2')
        if len(self.encoder_l1) != 16:
            warnings.warn(f'Expected 16 keys for each encoder layer, got {len(self.encoder_l1)}')
        if len(self.decoder_l1) != 26:
            warnings.warn(f'Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}')
        if len(self.decoder_l2) != 26:
            warnings.warn(f'Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}')

    @property
    def extra_keys(self):
        extra = []
        for k in self.state_keys:
            if k.startswith('encoder_l') or k.startswith('decoder_l') or k in [CONFIG_KEY, 'Wemb', 'encoder_Wemb', 'decoder_Wemb', 'Wpos', 'decoder_ff_logit_out_b']:
                continue
            else:
                extra.append(k)
        return extra

    def sub_keys(self, layer_prefix):
        return [remove_prefix(k, layer_prefix) for k in self.state_dict if k.startswith(layer_prefix)]

    def load_tokenizer(self):
        add_special_tokens_to_vocab(self.source_dir, not self.share_encoder_decoder_embeddings)
        return MarianTokenizer.from_pretrained(str(self.source_dir))

    def load_marian_model(self) -> MarianMTModel:
        state_dict, cfg = (self.state_dict, self.hf_config)
        if not cfg.static_position_embeddings:
            raise ValueError('config.static_position_embeddings should be True')
        model = MarianMTModel(cfg)
        if 'hidden_size' in cfg.to_dict():
            raise ValueError('hidden_size is in config')
        load_layers_(model.model.encoder.layers, state_dict, BART_CONVERTER)
        load_layers_(model.model.decoder.layers, state_dict, BART_CONVERTER, is_decoder=True)
        if self.cfg['tied-embeddings-src']:
            wemb_tensor = nn.Parameter(torch.FloatTensor(self.wemb))
            bias_tensor = nn.Parameter(torch.FloatTensor(self.final_bias))
            model.model.shared.weight = wemb_tensor
            model.model.encoder.embed_tokens = model.model.decoder.embed_tokens = model.model.shared
        else:
            wemb_tensor = nn.Parameter(torch.FloatTensor(self.wemb))
            model.model.encoder.embed_tokens.weight = wemb_tensor
            decoder_wemb_tensor = nn.Parameter(torch.FloatTensor(self.dec_wemb))
            bias_tensor = nn.Parameter(torch.FloatTensor(self.final_bias))
            model.model.decoder.embed_tokens.weight = decoder_wemb_tensor
        model.final_logits_bias = bias_tensor
        if 'Wpos' in state_dict:
            print('Unexpected: got Wpos')
            wpos_tensor = torch.tensor(state_dict['Wpos'])
            model.model.encoder.embed_positions.weight = wpos_tensor
            model.model.decoder.embed_positions.weight = wpos_tensor
        if cfg.normalize_embedding:
            if 'encoder_emb_ln_scale_pre' not in state_dict:
                raise ValueError('encoder_emb_ln_scale_pre is not in state dictionary')
            raise NotImplementedError('Need to convert layernorm_embedding')
        if self.extra_keys:
            raise ValueError(f'Failed to convert {self.extra_keys}')
        if model.get_input_embeddings().padding_idx != self.pad_token_id:
            raise ValueError(f'Padding tokens {model.get_input_embeddings().padding_idx} and {self.pad_token_id} mismatched')
        return model