from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_xlnet import XLNetConfig
@keras_serializable
class TFXLNetMainLayer(keras.layers.Layer):
    config_class = XLNetConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.return_dict = config.return_dict
        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer
        self.use_bfloat16 = config.use_bfloat16
        self.initializer_range = config.initializer_range
        self.word_embedding = TFSharedEmbeddings(config.vocab_size, config.d_model, initializer_range=config.initializer_range, name='word_embedding')
        self.layer = [TFXLNetLayer(config, name=f'layer_._{i}') for i in range(config.n_layer)]
        self.dropout = keras.layers.Dropout(config.dropout)
        self.use_mems_eval = config.use_mems_eval
        self.use_mems_train = config.use_mems_train

    def get_input_embeddings(self):
        return self.word_embedding

    def set_input_embeddings(self, value):
        self.word_embedding.weight = value
        self.word_embedding.vocab_size = shape_list(value)[0]

    def build(self, input_shape=None):
        initializer = get_initializer(self.initializer_range)
        self.mask_emb = self.add_weight(shape=(1, 1, self.d_model), initializer=initializer, trainable=True, name='mask_emb')
        if self.built:
            return
        self.built = True
        if getattr(self, 'word_embedding', None) is not None:
            with tf.name_scope(self.word_embedding.name):
                self.word_embedding.build(None)
        if getattr(self, 'layer', None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def create_mask(self, qlen, mlen):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.

        Args:
            qlen: TODO Lysandre didn't fill
            mlen: TODO Lysandre didn't fill

        ```

                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]
        ```
        """
        attn_mask = tf.ones([qlen, qlen])
        mask_u = tf.linalg.band_part(attn_mask, 0, -1)
        mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
        attn_mask_pad = tf.zeros([qlen, mlen])
        ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
        if self.same_length:
            mask_l = tf.linalg.band_part(attn_mask, -1, 0)
            ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
        return ret

    def cache_mem(self, curr_out, prev_mem):
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[:self.reuse_len]
        if self.mem_len is None or self.mem_len == 0:
            cutoff = 0
        else:
            cutoff = -self.mem_len
        if prev_mem is None:
            new_mem = curr_out[cutoff:]
        else:
            new_mem = tf.concat([prev_mem, curr_out], 0)[cutoff:]
        return tf.stop_gradient(new_mem)

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        sinusoid_inp = tf.einsum('i,d->id', pos_seq, inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], axis=-1)
        pos_emb = pos_emb[:, None, :]
        if bsz is not None:
            pos_emb = tf.tile(pos_emb, [1, bsz, 1])
        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        """create relative positional encoding."""
        freq_seq = tf.range(0, self.d_model, 2.0)
        inv_freq = 1 / 10000 ** (freq_seq / self.d_model)
        if self.attn_type == 'bi':
            beg, end = (klen, -qlen)
        elif self.attn_type == 'uni':
            beg, end = (klen, -1)
        else:
            raise ValueError(f'Unknown `attn_type` {self.attn_type}.')
        if self.bi_data:
            fwd_pos_seq = tf.range(beg, end, -1.0)
            bwd_pos_seq = tf.range(-beg, -end, 1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
                bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -self.clamp_len, self.clamp_len)
            if bsz is not None:
                if bsz % 2 != 0:
                    raise ValueError(f'With bi_data, the batch size {bsz} should be divisible by 2')
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)
            pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
        else:
            fwd_pos_seq = tf.range(beg, end, -1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)
        return pos_emb

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, mems: np.ndarray | tf.Tensor | None=None, perm_mask: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, token_type_ids: np.ndarray | tf.Tensor | None=None, input_mask: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, use_mems: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False):
        if training and use_mems is None:
            use_mems = self.use_mems_train
        else:
            use_mems = self.use_mems_eval
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_ids = tf.transpose(input_ids, perm=(1, 0))
            qlen, bsz = shape_list(input_ids)[:2]
        elif inputs_embeds is not None:
            inputs_embeds = tf.transpose(inputs_embeds, perm=(1, 0, 2))
            qlen, bsz = shape_list(inputs_embeds)[:2]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        token_type_ids = tf.transpose(token_type_ids, perm=(1, 0)) if token_type_ids is not None else None
        input_mask = tf.transpose(input_mask, perm=(1, 0)) if input_mask is not None else None
        attention_mask = tf.transpose(attention_mask, perm=(1, 0)) if attention_mask is not None else None
        perm_mask = tf.transpose(perm_mask, perm=(1, 2, 0)) if perm_mask is not None else None
        target_mapping = tf.transpose(target_mapping, perm=(1, 2, 0)) if target_mapping is not None else None
        mlen = shape_list(mems[0])[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen
        if self.attn_type == 'uni':
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError(f'Unsupported attention type: {self.attn_type}')
        assert input_mask is None or attention_mask is None, 'You can only use one of input_mask (uses 1 for padding) or attention_mask (uses 0 for padding, added for compatibility with BERT). Please choose one.'
        if input_mask is None and attention_mask is not None:
            one_cst = tf.constant(1.0)
            input_mask = 1.0 - tf.cast(attention_mask, dtype=one_cst.dtype)
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None
        if data_mask is not None:
            if mlen > 0:
                mems_mask = tf.zeros([shape_list(data_mask)[0], mlen, bsz])
                data_mask = tf.concat([mems_mask, data_mask], axis=1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]
        if attn_mask is not None:
            attn_mask = tf.cast(attn_mask > 0, dtype=attn_mask.dtype)
        if attn_mask is not None:
            non_tgt_mask = -tf.eye(qlen)
            if mlen > 0:
                non_tgt_mask = tf.concat([tf.zeros([qlen, mlen]), non_tgt_mask], axis=-1)
            non_tgt_mask = tf.cast(attn_mask + non_tgt_mask[:, :, None, None] > 0, dtype=non_tgt_mask.dtype)
        else:
            non_tgt_mask = None
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            check_embeddings_within_bounds(input_ids, self.word_embedding.vocab_size)
            word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k, training=training)
        if target_mapping is not None:
            word_emb_q = tf.tile(self.mask_emb, [shape_list(target_mapping)[0], bsz, 1])
            output_g = self.dropout(word_emb_q, training=training)
        else:
            output_g = None
        if token_type_ids is not None:
            if mlen > 0:
                mem_pad = tf.zeros([mlen, bsz], dtype=token_type_ids.dtype)
                cat_ids = tf.concat([mem_pad, token_type_ids], 0)
            else:
                cat_ids = token_type_ids
            seg_mat = tf.cast(tf.logical_not(tf.equal(token_type_ids[:, None], cat_ids[None, :])), dtype=token_type_ids.dtype)
            seg_mat = tf.one_hot(seg_mat, 2)
        else:
            seg_mat = None
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb, training=training)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.n_layer
        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)
        attentions = [] if output_attentions else None
        hidden_states = [] if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            if use_mems:
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)
            outputs = layer_module(output_h, output_g, non_tgt_mask, attn_mask, pos_emb, seg_mat, mems[i], target_mapping, head_mask[i], output_attentions, training=training)
            output_h, output_g = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])
        if output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)
        output = self.dropout(output_g if output_g is not None else output_h, training=training)
        output = tf.transpose(output, perm=(1, 0, 2))
        if not use_mems:
            new_mems = None
        if output_hidden_states:
            if output_g is not None:
                hidden_states = tuple((tf.transpose(h, perm=(1, 0, 2)) for hs in hidden_states for h in hs))
            else:
                hidden_states = tuple((tf.transpose(hs, perm=(1, 0, 2)) for hs in hidden_states))
        if output_attentions:
            if target_mapping is not None:
                attentions = tuple((tuple((tf.transpose(attn_stream, perm=(2, 3, 0, 1)) for attn_stream in t)) for t in attentions))
            else:
                attentions = tuple((tf.transpose(t, perm=(2, 3, 0, 1)) for t in attentions))
        if not return_dict:
            return tuple((v for v in [output, new_mems, hidden_states, attentions] if v is not None))
        return TFXLNetModelOutput(last_hidden_state=output, mems=new_mems, hidden_states=hidden_states, attentions=attentions)