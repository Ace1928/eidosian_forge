from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ....modeling_tf_utils import (
from ....tf_utils import shape_list, stable_softmax
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_tf_transfo_xl_utilities import TFAdaptiveSoftmaxMask
@add_start_docstrings('\n    The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive\n    input embeddings)\n    ', TRANSFO_XL_START_DOCSTRING)
class TFTransfoXLLMHeadModel(TFTransfoXLPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TFTransfoXLMainLayer(config, name='transformer')
        self.sample_softmax = config.sample_softmax
        assert self.sample_softmax <= 0, 'Sampling from the softmax is not implemented yet. Please look at issue: #3310: https://github.com/huggingface/transformers/issues/3310'
        self.crit = TFAdaptiveSoftmaxMask(config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val, name='crit')

    def _resize_token_embeddings(self, new_num_tokens):
        raise NotImplementedError()

    def get_output_embeddings(self):
        """Double-check if you are using adaptive softmax."""
        if len(self.crit.out_layers) > 0:
            return self.crit.out_layers[-1]
        return None

    def reset_memory_length(self, mem_len):
        self.transformer.reset_memory_length(mem_len)

    def init_mems(self, bsz):
        return self.transformer.init_mems(bsz)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFTransfoXLLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, mems: List[tf.Tensor] | None=None, head_mask: np.ndarray | tf.Tensor | None=None, inputs_embeds: np.ndarray | tf.Tensor | None=None, output_attentions: bool | None=None, output_hidden_states: bool | None=None, return_dict: bool | None=None, labels: np.ndarray | tf.Tensor | None=None, training: bool=False) -> TFTransfoXLLMHeadModelOutput | Tuple[tf.Tensor]:
        if input_ids is not None:
            bsz, tgt_len = shape_list(input_ids)[:2]
        else:
            bsz, tgt_len = shape_list(inputs_embeds)[:2]
        transformer_outputs = self.transformer(input_ids, mems, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict, training=training)
        last_hidden = transformer_outputs[0]
        pred_hid = last_hidden[:, -tgt_len:]
        softmax_output = self.crit(pred_hid, labels, training=training)
        prediction_scores = softmax_output if labels is None else ()
        if not return_dict:
            return (prediction_scores,) + transformer_outputs[1:]
        return TFTransfoXLLMHeadModelOutput(prediction_scores=prediction_scores, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **model_kwargs):
        inputs = {}
        if past_key_values:
            input_ids = tf.expand_dims(input_ids[:, -1], axis=-1)
        else:
            input_ids = input_ids
        return inputs

    def tf_to_pt_weight_rename(self, tf_weight):
        if self.config.tie_word_embeddings and 'crit.out_layers' in tf_weight:
            return (tf_weight, tf_weight.replace('crit.out_layers', 'transformer.word_emb.emb_layers'))
        elif self.config.tie_projs and 'crit.out_projs' in tf_weight:
            for i, tie_proj in enumerate(self.config.tie_projs):
                if tie_proj and self.config.div_val == 1 and (self.config.d_model != self.config.d_embed):
                    return (tf_weight, tf_weight.replace(f'crit.out_projs.{i}', 'transformer.word_emb.emb_projs.0'))
                elif tie_proj and self.config.div_val != 1:
                    return (tf_weight, tf_weight.replace('crit.out_projs', 'transformer.word_emb.emb_projs'))
        else:
            return (tf_weight,)