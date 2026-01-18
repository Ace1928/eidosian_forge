import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_lxmert import LxmertConfig
@add_start_docstrings('Lxmert Model with a specified pretraining head on top.', LXMERT_START_DOCSTRING)
class LxmertForPreTraining(LxmertPreTrainedModel):
    _tied_weights_keys = ['cls.predictions.decoder.weight']

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_qa_labels = config.num_qa_labels
        self.visual_loss_normalizer = config.visual_loss_normalizer
        self.task_mask_lm = config.task_mask_lm
        self.task_obj_predict = config.task_obj_predict
        self.task_matched = config.task_matched
        self.task_qa = config.task_qa
        self.lxmert = LxmertModel(config)
        self.cls = LxmertPreTrainingHeads(config, self.lxmert.embeddings.word_embeddings.weight)
        if self.task_obj_predict:
            self.obj_predict_head = LxmertVisualObjHead(config)
        if self.task_qa:
            self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)
        self.post_init()
        self.loss_fcts = {'l2': SmoothL1Loss(reduction='none'), 'visual_ce': CrossEntropyLoss(reduction='none'), 'ce': CrossEntropyLoss()}
        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses['obj'] = {'shape': (-1,), 'num': config.num_object_labels, 'loss': 'visual_ce'}
        if config.visual_attr_loss:
            visual_losses['attr'] = {'shape': (-1,), 'num': config.num_attr_labels, 'loss': 'visual_ce'}
        if config.visual_feat_loss:
            visual_losses['feat'] = {'shape': (-1, config.visual_feat_dim), 'num': config.visual_feat_dim, 'loss': 'l2'}
        self.visual_losses = visual_losses

    def resize_num_qa_labels(self, num_labels):
        """
        Build a resized question answering linear layer Module from a provided new linear layer. Increasing the size
        will add newly initialized weights. Reducing the size will remove weights from the end

        Args:
            num_labels (`int`, *optional*):
                New number of labels in the linear layer weight matrix. Increasing the size will add newly initialized
                weights at the end. Reducing the size will remove weights from the end. If not provided or `None`, just
                returns a pointer to the qa labels ``torch.nn.Linear``` module of the model without doing anything.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear layer or the old Linear layer
        """
        cur_qa_logit_layer = self.get_qa_logit_layer()
        if num_labels is None or cur_qa_logit_layer is None:
            return
        new_qa_logit_layer = self._resize_qa_labels(num_labels)
        self.config.num_qa_labels = num_labels
        self.num_qa_labels = num_labels
        return new_qa_logit_layer

    def _resize_qa_labels(self, num_labels):
        cur_qa_logit_layer = self.get_qa_logit_layer()
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
        self._set_qa_logit_layer(new_qa_logit_layer)
        return self.get_qa_logit_layer()

    def get_qa_logit_layer(self) -> nn.Module:
        """
        Returns the linear layer that produces question answering logits.

        Returns:
            `nn.Module`: A torch module mapping the question answering prediction hidden states or `None` if LXMERT
            does not have a visual answering head.
        """
        if hasattr(self, 'answer_head'):
            return self.answer_head.logit_fc[-1]

    def _set_qa_logit_layer(self, qa_logit_layer):
        self.answer_head.logit_fc[-1] = qa_logit_layer

    def _get_resized_qa_labels(self, cur_qa_logit_layer, num_labels):
        if num_labels is None:
            return cur_qa_logit_layer
        cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.size()
        if cur_qa_labels == num_labels:
            return cur_qa_logit_layer
        if getattr(cur_qa_logit_layer, 'bias', None) is not None:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels)
        else:
            new_qa_logit_layer = nn.Linear(hidden_dim, num_labels, bias=False)
        new_qa_logit_layer.to(cur_qa_logit_layer.weight.device)
        self._init_weights(new_qa_logit_layer)
        num_labels_to_copy = min(cur_qa_labels, num_labels)
        new_qa_logit_layer.weight.data[:num_labels_to_copy, :] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
        if getattr(cur_qa_logit_layer, 'bias', None) is not None:
            new_qa_logit_layer.bias.data[:num_labels_to_copy] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]
        return new_qa_logit_layer

    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=LxmertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, visual_feats: Optional[torch.FloatTensor]=None, visual_pos: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, visual_attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, obj_labels: Optional[Dict[str, Tuple[torch.FloatTensor, torch.FloatTensor]]]=None, matched_label: Optional[torch.LongTensor]=None, ans: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[LxmertForPreTrainingOutput, Tuple[torch.FloatTensor]]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        obj_labels (`Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]`, *optional*):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            `(batch_size, num_features)` and `(batch_size, num_features, visual_feature_dim)` for each the label id and
            the label score respectively
        matched_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans (`Torch.Tensor` of shape `(batch_size)`, *optional*):
            a one hot representation hof the correct answer *optional*

        Returns:
        """
        if 'masked_lm_labels' in kwargs:
            warnings.warn('The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.', FutureWarning)
            labels = kwargs.pop('masked_lm_labels')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        lxmert_output = self.lxmert(input_ids=input_ids, visual_feats=visual_feats, visual_pos=visual_pos, token_type_ids=token_type_ids, attention_mask=attention_mask, visual_attention_mask=visual_attention_mask, inputs_embeds=inputs_embeds, output_hidden_states=output_hidden_states, output_attentions=output_attentions, return_dict=return_dict)
        lang_output, visual_output, pooled_output = (lxmert_output[0], lxmert_output[1], lxmert_output[2])
        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)
        if self.task_qa:
            answer_score = self.answer_head(pooled_output)
        else:
            answer_score = pooled_output[0][0]
        total_loss = None if labels is None and matched_label is None and (obj_labels is None) and (ans is None) else torch.tensor(0.0, device=device)
        if labels is not None and self.task_mask_lm:
            masked_lm_loss = self.loss_fcts['ce'](lang_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss += masked_lm_loss
        if matched_label is not None and self.task_matched:
            matched_loss = self.loss_fcts['ce'](cross_relationship_score.view(-1, 2), matched_label.view(-1))
            total_loss += matched_loss
        if obj_labels is not None and self.task_obj_predict:
            total_visual_loss = torch.tensor(0.0, device=input_ids.device)
            visual_prediction_scores_dict = self.obj_predict_head(visual_output)
            for key, key_info in self.visual_losses.items():
                label, mask_conf = obj_labels[key]
                output_dim = key_info['num']
                loss_fct_name = key_info['loss']
                label_shape = key_info['shape']
                weight = self.visual_loss_normalizer
                visual_loss_fct = self.loss_fcts[loss_fct_name]
                visual_prediction_scores = visual_prediction_scores_dict[key]
                visual_loss = visual_loss_fct(visual_prediction_scores.view(-1, output_dim), label.view(label_shape))
                if visual_loss.dim() > 1:
                    visual_loss = visual_loss.mean(1)
                visual_loss = (visual_loss * mask_conf.view(-1)).mean() * weight
                total_visual_loss += visual_loss
            total_loss += total_visual_loss
        if ans is not None and self.task_qa:
            answer_loss = self.loss_fcts['ce'](answer_score.view(-1, self.num_qa_labels), ans.view(-1))
            total_loss += answer_loss
        if not return_dict:
            output = (lang_prediction_scores, cross_relationship_score, answer_score) + lxmert_output[3:]
            return (total_loss,) + output if total_loss is not None else output
        return LxmertForPreTrainingOutput(loss=total_loss, prediction_logits=lang_prediction_scores, cross_relationship_score=cross_relationship_score, question_answering_score=answer_score, language_hidden_states=lxmert_output.language_hidden_states, vision_hidden_states=lxmert_output.vision_hidden_states, language_attentions=lxmert_output.language_attentions, vision_attentions=lxmert_output.vision_attentions, cross_encoder_attentions=lxmert_output.cross_encoder_attentions)