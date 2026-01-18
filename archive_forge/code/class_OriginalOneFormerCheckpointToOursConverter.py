import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn
from transformers import CLIPTokenizer, DinatConfig, SwinConfig
from transformers.models.oneformer.image_processing_oneformer import OneFormerImageProcessor
from transformers.models.oneformer.modeling_oneformer import (
from transformers.models.oneformer.processing_oneformer import OneFormerProcessor
from transformers.utils import logging
class OriginalOneFormerCheckpointToOursConverter:

    def __init__(self, original_model: nn.Module, config: OneFormerConfig):
        self.original_model = original_model
        self.config = config

    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        for src_key, dst_key in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)

    def replace_swin_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: OneFormerConfig):
        dst_prefix: str = 'pixel_level_module.encoder'
        src_prefix: str = 'backbone'
        renamed_keys = [(f'{src_prefix}.patch_embed.proj.weight', f'{dst_prefix}.embeddings.patch_embeddings.projection.weight'), (f'{src_prefix}.patch_embed.proj.bias', f'{dst_prefix}.embeddings.patch_embeddings.projection.bias'), (f'{src_prefix}.patch_embed.norm.weight', f'{dst_prefix}.embeddings.norm.weight'), (f'{src_prefix}.patch_embed.norm.bias', f'{dst_prefix}.embeddings.norm.bias')]
        num_layers = len(config.backbone_config.depths)
        for layer_idx in range(num_layers):
            for block_idx in range(config.backbone_config.depths[layer_idx]):
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.bias', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.bias'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_bias_table', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_bias_table')])
                src_att_weight = src_state_dict[f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight']
                src_att_bias = src_state_dict[f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias']
                size = src_att_weight.shape[0]
                offset = size // 3
                dst_state_dict[f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.weight'] = src_att_weight[:offset, :]
                dst_state_dict[f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.bias'] = src_att_bias[:offset]
                dst_state_dict[f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.weight'] = src_att_weight[offset:offset * 2, :]
                dst_state_dict[f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.bias'] = src_att_bias[offset:offset * 2]
                dst_state_dict[f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.weight'] = src_att_weight[-offset:, :]
                dst_state_dict[f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.bias'] = src_att_bias[-offset:]
                src_state_dict.pop(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight')
                src_state_dict.pop(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias')
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.bias', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.bias')])
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.bias', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.bias')])
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.bias', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.bias'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.bias', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.bias')])
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_index', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_index')])
            if layer_idx < num_layers - 1:
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.downsample.reduction.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.downsample.reduction.weight'), (f'{src_prefix}.layers.{layer_idx}.downsample.norm.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.downsample.norm.weight'), (f'{src_prefix}.layers.{layer_idx}.downsample.norm.bias', f'{dst_prefix}.encoder.layers.{layer_idx}.downsample.norm.bias')])
            renamed_keys.extend([(f'{src_prefix}.norm{layer_idx}.weight', f'{dst_prefix}.hidden_states_norms.stage{layer_idx + 1}.weight'), (f'{src_prefix}.norm{layer_idx}.bias', f'{dst_prefix}.hidden_states_norms.stage{layer_idx + 1}.bias')])
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_dinat_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: OneFormerConfig):
        dst_prefix: str = 'pixel_level_module.encoder'
        src_prefix: str = 'backbone'

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [(f'{src_prefix}.weight', f'{dst_prefix}.weight'), (f'{src_prefix}.bias', f'{dst_prefix}.bias')]
        renamed_keys = rename_keys_for_weight_bias(f'{src_prefix}.patch_embed.norm', f'{dst_prefix}.embeddings.norm')
        for i in range(2):
            renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.patch_embed.proj.{i}', f'{dst_prefix}.embeddings.patch_embeddings.projection.{i}'))
        num_layers = len(config.backbone_config.depths)
        for layer_idx in range(num_layers):
            for block_idx in range(config.backbone_config.depths[layer_idx]):
                renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.norm1', f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.layernorm_before'))
                renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.norm2', f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.layernorm_after'))
                renamed_keys.extend([(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.rpb', f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.rpb')])
                src_att_weight = src_state_dict[f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.qkv.weight']
                src_att_bias = src_state_dict[f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.qkv.bias']
                size = src_att_weight.shape[0]
                offset = size // 3
                dst_state_dict[f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.query.weight'] = src_att_weight[:offset, :]
                dst_state_dict[f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.query.bias'] = src_att_bias[:offset]
                dst_state_dict[f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.key.weight'] = src_att_weight[offset:offset * 2, :]
                dst_state_dict[f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.key.bias'] = src_att_bias[offset:offset * 2]
                dst_state_dict[f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.value.weight'] = src_att_weight[-offset:, :]
                dst_state_dict[f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.self.value.bias'] = src_att_bias[-offset:]
                src_state_dict.pop(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.qkv.weight')
                src_state_dict.pop(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.qkv.bias')
                renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.attn.proj', f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.attention.output.dense'))
                renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.mlp.fc1', f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.intermediate.dense'))
                renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.levels.{layer_idx}.blocks.{block_idx}.mlp.fc2', f'{dst_prefix}.encoder.levels.{layer_idx}.layers.{block_idx}.output.dense'))
            if layer_idx < num_layers - 1:
                renamed_keys.extend([(f'{src_prefix}.levels.{layer_idx}.downsample.reduction.weight', f'{dst_prefix}.encoder.levels.{layer_idx}.downsample.reduction.weight'), (f'{src_prefix}.levels.{layer_idx}.downsample.norm.weight', f'{dst_prefix}.encoder.levels.{layer_idx}.downsample.norm.weight'), (f'{src_prefix}.levels.{layer_idx}.downsample.norm.bias', f'{dst_prefix}.encoder.levels.{layer_idx}.downsample.norm.bias')])
            renamed_keys.extend([(f'{src_prefix}.norm{layer_idx}.weight', f'{dst_prefix}.hidden_states_norms.stage{layer_idx + 1}.weight'), (f'{src_prefix}.norm{layer_idx}.bias', f'{dst_prefix}.hidden_states_norms.stage{layer_idx + 1}.bias')])
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_pixel_module(self, dst_state_dict: StateDict, src_state_dict: StateDict, is_swin: bool):
        dst_prefix: str = 'pixel_level_module.decoder'
        src_prefix: str = 'sem_seg_head.pixel_decoder'
        if is_swin:
            self.replace_swin_backbone(dst_state_dict, src_state_dict, self.config)
        else:
            self.replace_dinat_backbone(dst_state_dict, src_state_dict, self.config)

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [(f'{src_prefix}.weight', f'{dst_prefix}.weight'), (f'{src_prefix}.bias', f'{dst_prefix}.bias')]

        def rename_keys_for_self_attn(src_prefix: str, dst_prefix: str):
            self_attn_keys = []
            self_attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.attention_weights', f'{dst_prefix}.attention_weights'))
            self_attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.output_proj', f'{dst_prefix}.output_proj'))
            self_attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.sampling_offsets', f'{dst_prefix}.sampling_offsets'))
            self_attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.value_proj', f'{dst_prefix}.value_proj'))
            return self_attn_keys

        def rename_keys_for_encoder_layer(src_prefix: str, dst_prefix: str):
            encoder_keys = []
            encoder_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear1', f'{dst_prefix}.fc1'))
            encoder_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear2', f'{dst_prefix}.fc2'))
            encoder_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm1', f'{dst_prefix}.self_attn_layer_norm'))
            encoder_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm2', f'{dst_prefix}.final_layer_norm'))
            encoder_keys.extend(rename_keys_for_self_attn(f'{src_prefix}.self_attn', f'{dst_prefix}.self_attn'))
            return encoder_keys
        renamed_keys = [(f'{src_prefix}.adapter_1.weight', f'{dst_prefix}.adapter_1.0.weight'), (f'{src_prefix}.adapter_1.norm.weight', f'{dst_prefix}.adapter_1.1.weight'), (f'{src_prefix}.adapter_1.norm.bias', f'{dst_prefix}.adapter_1.1.bias')]
        renamed_keys.extend([(f'{src_prefix}.layer_1.weight', f'{dst_prefix}.layer_1.0.weight'), (f'{src_prefix}.layer_1.norm.weight', f'{dst_prefix}.layer_1.1.weight'), (f'{src_prefix}.layer_1.norm.bias', f'{dst_prefix}.layer_1.1.bias')])
        for i in range(3):
            for j in range(2):
                renamed_keys.extend([(f'{src_prefix}.input_proj.{i}.{j}.weight', f'{dst_prefix}.input_projections.{i}.{j}.weight'), (f'{src_prefix}.input_proj.{i}.{j}.bias', f'{dst_prefix}.input_projections.{i}.{j}.bias')])
        renamed_keys.extend([(f'{src_prefix}.transformer.level_embed', f'{dst_prefix}.level_embed')])
        for layer_idx in range(self.config.encoder_layers):
            renamed_keys.extend(rename_keys_for_encoder_layer(f'{src_prefix}.transformer.encoder.layers.{layer_idx}', f'{dst_prefix}.encoder.layers.{layer_idx}'))
        renamed_keys.extend([(f'{src_prefix}.mask_features.weight', f'{dst_prefix}.mask_projection.weight'), (f'{src_prefix}.mask_features.bias', f'{dst_prefix}.mask_projection.bias')])
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_keys_qkv_transformer_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'transformer_module.decoder.layers'
        src_prefix: str = 'sem_seg_head.predictor'
        for i in range(self.config.decoder_layers - 1):
            in_proj_weight = src_state_dict.pop(f'{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_weight')
            in_proj_bias = src_state_dict.pop(f'{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_bias')
            dst_state_dict[f'{dst_prefix}.{i}.self_attn.self_attn.q_proj.weight'] = in_proj_weight[:256, :]
            dst_state_dict[f'{dst_prefix}.{i}.self_attn.self_attn.q_proj.bias'] = in_proj_bias[:256]
            dst_state_dict[f'{dst_prefix}.{i}.self_attn.self_attn.k_proj.weight'] = in_proj_weight[256:512, :]
            dst_state_dict[f'{dst_prefix}.{i}.self_attn.self_attn.k_proj.bias'] = in_proj_bias[256:512]
            dst_state_dict[f'{dst_prefix}.{i}.self_attn.self_attn.v_proj.weight'] = in_proj_weight[-256:, :]
            dst_state_dict[f'{dst_prefix}.{i}.self_attn.self_attn.v_proj.bias'] = in_proj_bias[-256:]

    def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'transformer_module'
        src_prefix: str = 'sem_seg_head.predictor'

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [(f'{src_prefix}.weight', f'{dst_prefix}.weight'), (f'{src_prefix}.bias', f'{dst_prefix}.bias')]

        def rename_keys_for_attn(src_prefix: str, dst_prefix: str):
            attn_keys = [(f'{src_prefix}.in_proj_bias', f'{dst_prefix}.in_proj_bias'), (f'{src_prefix}.in_proj_weight', f'{dst_prefix}.in_proj_weight')]
            attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.out_proj', f'{dst_prefix}.out_proj'))
            return attn_keys

        def rename_keys_for_self_attn(src_prefix: str, dst_prefix: str):
            attn_keys = []
            attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.out_proj', f'{dst_prefix}.out_proj'))
            return attn_keys

        def rename_keys_for_query_transformer_layer(src_prefix: str, dst_prefix: str):
            query_transformer_layer_keys = []
            query_transformer_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear1', f'{dst_prefix}.linear1'))
            query_transformer_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear2', f'{dst_prefix}.linear2'))
            query_transformer_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm1', f'{dst_prefix}.norm1'))
            query_transformer_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm2', f'{dst_prefix}.norm2'))
            query_transformer_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm3', f'{dst_prefix}.norm3'))
            query_transformer_layer_keys.extend(rename_keys_for_attn(f'{src_prefix}.self_attn', f'{dst_prefix}.self_attn'))
            query_transformer_layer_keys.extend(rename_keys_for_attn(f'{src_prefix}.multihead_attn', f'{dst_prefix}.multihead_attn'))
            return query_transformer_layer_keys

        def rename_keys_for_cross_attn_layer(src_prefix: str, dst_prefix: str):
            cross_attn_layer_keys = []
            cross_attn_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm', f'{dst_prefix}.norm'))
            cross_attn_layer_keys.extend(rename_keys_for_attn(f'{src_prefix}.multihead_attn', f'{dst_prefix}.multihead_attn'))
            return cross_attn_layer_keys

        def rename_keys_for_self_attn_layer(src_prefix: str, dst_prefix: str):
            self_attn_layer_keys = []
            self_attn_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm', f'{dst_prefix}.norm'))
            self_attn_layer_keys.extend(rename_keys_for_self_attn(f'{src_prefix}.self_attn', f'{dst_prefix}.self_attn'))
            return self_attn_layer_keys

        def rename_keys_for_ffn_layer(src_prefix: str, dst_prefix: str):
            ffn_layer_keys = []
            ffn_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear1', f'{dst_prefix}.linear1'))
            ffn_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear2', f'{dst_prefix}.linear2'))
            ffn_layer_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm', f'{dst_prefix}.norm'))
            return ffn_layer_keys

        def rename_keys_for_transformer_decoder_layer(src_prefix: str, dst_prefix: str, idx: int):
            transformer_decoder_layer_keys = []
            transformer_decoder_layer_keys.extend(rename_keys_for_cross_attn_layer(f'{src_prefix}.transformer_cross_attention_layers.{idx}', f'{dst_prefix}.{idx}.cross_attn'))
            transformer_decoder_layer_keys.extend(rename_keys_for_self_attn_layer(f'{src_prefix}.transformer_self_attention_layers.{idx}', f'{dst_prefix}.{idx}.self_attn'))
            transformer_decoder_layer_keys.extend(rename_keys_for_ffn_layer(f'{src_prefix}.transformer_ffn_layers.{idx}', f'{dst_prefix}.{idx}.ffn'))
            return transformer_decoder_layer_keys
        renamed_keys = [(f'{src_prefix}.query_embed.weight', f'{dst_prefix}.queries_embedder.weight'), (f'{src_prefix}.level_embed.weight', f'{dst_prefix}.level_embed.weight')]
        renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.decoder_norm', f'{dst_prefix}.decoder.decoder_norm'))
        renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.class_input_proj', f'{dst_prefix}.decoder.query_input_projection'))
        renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.class_embed', f'{dst_prefix}.decoder.class_embed'))
        for i in range(3):
            renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.mask_embed.layers.{i}', f'{dst_prefix}.decoder.mask_embed.layers.{i}.0'))
        renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.class_transformer.decoder.norm', f'{dst_prefix}.decoder.query_transformer.decoder.norm'))
        for i in range(self.config.query_dec_layers):
            renamed_keys.extend(rename_keys_for_query_transformer_layer(f'{src_prefix}.class_transformer.decoder.layers.{i}', f'{dst_prefix}.decoder.query_transformer.decoder.layers.{i}'))
        for i in range(self.config.decoder_layers - 1):
            renamed_keys.extend(rename_keys_for_transformer_decoder_layer(f'{src_prefix}', f'{dst_prefix}.decoder.layers', i))
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
        self.replace_keys_qkv_transformer_decoder(dst_state_dict, src_state_dict)

    def replace_task_mlp(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'task_encoder'
        src_prefix: str = 'task_mlp'

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [(f'{src_prefix}.weight', f'{dst_prefix}.weight'), (f'{src_prefix}.bias', f'{dst_prefix}.bias')]
        renamed_keys = []
        for i in range(2):
            renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.layers.{i}', f'{dst_prefix}.task_mlp.layers.{i}.0'))
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_text_projector(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'text_mapper.text_projector'
        src_prefix: str = 'text_projector'

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [(f'{src_prefix}.weight', f'{dst_prefix}.weight'), (f'{src_prefix}.bias', f'{dst_prefix}.bias')]
        renamed_keys = []
        for i in range(self.config.text_encoder_config['text_encoder_proj_layers']):
            renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.layers.{i}', f'{dst_prefix}.{i}.0'))
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_text_mapper(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'text_mapper.text_encoder'
        src_prefix: str = 'text_encoder'
        self.replace_text_projector(dst_state_dict, src_state_dict)

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [(f'{src_prefix}.weight', f'{dst_prefix}.weight'), (f'{src_prefix}.bias', f'{dst_prefix}.bias')]

        def rename_keys_for_attn(src_prefix: str, dst_prefix: str):
            attn_keys = [(f'{src_prefix}.in_proj_bias', f'{dst_prefix}.in_proj_bias'), (f'{src_prefix}.in_proj_weight', f'{dst_prefix}.in_proj_weight')]
            attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.out_proj', f'{dst_prefix}.out_proj'))
            return attn_keys

        def rename_keys_for_layer(src_prefix: str, dst_prefix: str):
            resblock_keys = []
            resblock_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.mlp.c_fc', f'{dst_prefix}.mlp.fc1'))
            resblock_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.mlp.c_proj', f'{dst_prefix}.mlp.fc2'))
            resblock_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.ln_1', f'{dst_prefix}.layer_norm1'))
            resblock_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.ln_2', f'{dst_prefix}.layer_norm2'))
            resblock_keys.extend(rename_keys_for_attn(f'{src_prefix}.attn', f'{dst_prefix}.self_attn'))
            return resblock_keys
        renamed_keys = [('prompt_ctx.weight', 'text_mapper.prompt_ctx.weight')]
        renamed_keys.extend([(f'{src_prefix}.positional_embedding', f'{dst_prefix}.positional_embedding'), (f'{src_prefix}.token_embedding.weight', f'{dst_prefix}.token_embedding.weight')])
        renamed_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.ln_final', f'{dst_prefix}.ln_final'))
        for i in range(self.config.text_encoder_config['text_encoder_num_layers']):
            renamed_keys.extend(rename_keys_for_layer(f'{src_prefix}.transformer.resblocks.{i}', f'{dst_prefix}.transformer.layers.{i}'))
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def convert(self, oneformer: OneFormerModel, is_swin: bool) -> OneFormerModel:
        dst_state_dict = TrackedStateDict(oneformer.state_dict())
        src_state_dict = self.original_model.state_dict()
        self.replace_pixel_module(dst_state_dict, src_state_dict, is_swin)
        self.replace_transformer_module(dst_state_dict, src_state_dict)
        self.replace_task_mlp(dst_state_dict, src_state_dict)
        if self.config.is_training:
            self.replace_text_mapper(dst_state_dict, src_state_dict)
        logger.info(f'Missed keys are {pformat(dst_state_dict.diff())}')
        logger.info(f'Not copied keys are {pformat(src_state_dict.keys())}')
        logger.info('🙌 Done')
        oneformer.load_state_dict(dst_state_dict)
        return oneformer

    @staticmethod
    def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
        checkpoints: List[Path] = checkpoints_dir.glob('**/*.pth')
        for checkpoint in checkpoints:
            logger.info(f'💪 Converting {checkpoint.stem}')
            config: Path = config_dir / f'{checkpoint.stem}.yaml'
            yield (config, checkpoint)