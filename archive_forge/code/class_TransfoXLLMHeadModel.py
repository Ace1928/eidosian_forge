import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....modeling_utils import PreTrainedModel
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_transfo_xl_utilities import ProjectedAdaptiveLogSoftmax
@add_start_docstrings('\n    The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive\n    input embeddings)\n    ', TRANSFO_XL_START_DOCSTRING)
class TransfoXLLMHeadModel(TransfoXLPreTrainedModel):
    _tied_weights_keys = ['crit\\.out_projs\\.\\d+', 'crit\\.out_layers\\.\\d+\\.weight']

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TransfoXLModel(config)
        self.sample_softmax = config.sample_softmax
        self.trainer_compatible = getattr(config, 'trainer_compatible', False)
        if not self.trainer_compatible:
            warnings.warn('The output of TransfoXL will be updated in v5 to support a single loss as first argument. In order to use that updated output, please specify `trainer_compatible=True` as your configuration attribute.', DeprecationWarning)
        assert self.sample_softmax <= 0, 'Sampling from the softmax is not implemented yet. Please look at issue: #3310: https://github.com/huggingface/transformers/issues/3310'
        self.crit = ProjectedAdaptiveLogSoftmax(config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val)
        self.post_init()

    def tie_weights(self):
        """
        Run this to be sure output and input (adaptive) softmax weights are tied
        """
        if self.config.tie_word_embeddings:
            for i in range(len(self.crit.out_layers)):
                self._tie_or_clone_weights(self.crit.out_layers[i], self.transformer.word_emb.emb_layers[i])
        if self.config.tie_projs:
            for i, tie_proj in enumerate(self.config.tie_projs):
                if tie_proj and self.config.div_val == 1 and (self.config.d_model != self.config.d_embed):
                    if self.config.torchscript:
                        self.crit.out_projs[i] = nn.Parameter(self.transformer.word_emb.emb_projs[0].clone())
                    else:
                        self.crit.out_projs[i] = self.transformer.word_emb.emb_projs[0]
                elif tie_proj and self.config.div_val != 1:
                    if self.config.torchscript:
                        self.crit.out_projs[i] = nn.Parameter(self.transformer.word_emb.emb_projs[i].clone())
                    else:
                        self.crit.out_projs[i] = self.transformer.word_emb.emb_projs[i]

    def reset_memory_length(self, mem_len):
        self.transformer.reset_memory_length(mem_len)

    def init_mems(self, bsz):
        return self.transformer.init_mems(bsz)

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TransfoXLLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, mems: Optional[List[torch.FloatTensor]]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, TransfoXLLMHeadModelOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None:
            bsz, tgt_len = (input_ids.size(0), input_ids.size(1))
        elif inputs_embeds is not None:
            bsz, tgt_len = (inputs_embeds.size(0), inputs_embeds.size(1))
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        transformer_outputs = self.transformer(input_ids, mems=mems, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden = transformer_outputs[0]
        pred_hid = last_hidden[:, -tgt_len:]
        if labels is not None:
            miss_valid_label = labels[0, 1:].sum() == (labels.size(1) - 1) * -100
            if miss_valid_label:
                labels[0, 1] = self.config.eos_token_id
        softmax_output = self.crit(pred_hid, labels)
        prediction_scores = softmax_output.view(bsz, tgt_len, -1) if labels is None else ()
        if labels is not None:
            losses = softmax_output.view(bsz, tgt_len - 1)
            loss = losses[losses != 0].mean()
        else:
            losses, loss = (None, None)
        if not return_dict:
            if self.trainer_compatible:
                output = (prediction_scores, losses) if losses is not None else (prediction_scores,)
                output += transformer_outputs[1:]
                return (loss,) + output if loss is not None else output
            else:
                output = (prediction_scores, *transformer_outputs[1:])
                output = (losses,) + output if losses is not None else output
                return output + (loss,) if loss is not None else output
        return TransfoXLLMHeadModelOutput(loss=loss, prediction_scores=prediction_scores, losses=losses, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)

    def get_output_embeddings(self):
        """Double-check if you are using adaptive softmax."""
        if self.sample_softmax > 0:
            return self.out_layer
        else:
            return self.crit.out_layers[-1]

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **model_kwargs):
        inputs = {}
        if past_key_values:
            inputs['mems'] = past_key_values
            inputs['input_ids'] = input_ids[:, -1].unsqueeze(-1)
        else:
            inputs['input_ids'] = input_ids
        return inputs

    def _resize_cutoffs(self, new_num_tokens, new_emb_size, new_embedding_shapes, layer):
        new_cutoffs = super()._resize_cutoffs(new_num_tokens, new_emb_size, new_embedding_shapes, layer)
        self.crit.cutoffs = new_cutoffs
        self.crit.cutoff_ends = [0] + new_cutoffs
        self.crit.n_token = new_num_tokens

    @staticmethod
    def _reorder_cache(mems: List[torch.Tensor], beam_idx: torch.Tensor) -> List[torch.Tensor]:
        """
        This function is used to re-order the `mems` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `mems` with the correct beam_idx at every
        generation step.
        """
        return [layer_past.index_select(1, beam_idx.to(layer_past.device)) for layer_past in mems]