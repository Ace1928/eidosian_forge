from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_lxmert import LxmertConfig
@add_start_docstrings('Lxmert Model with a `language modeling` head on top.', LXMERT_START_DOCSTRING)
class TFLxmertForPreTraining(TFLxmertPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.num_qa_labels = config.num_qa_labels
        self.visual_loss_normalizer = config.visual_loss_normalizer
        self.task_mask_lm = config.task_mask_lm
        self.task_obj_predict = config.task_obj_predict
        self.task_matched = config.task_matched
        self.task_qa = config.task_qa
        self.lxmert = TFLxmertMainLayer(config, name='lxmert')
        self.cls = TFLxmertPreTrainingHeads(config, self.lxmert.embeddings, name='cls')
        if self.task_obj_predict:
            self.obj_predict_head = TFLxmertVisualObjHead(config, name='obj_predict_head')
        if self.task_qa:
            self.answer_head = TFLxmertVisualAnswerHead(config, self.num_qa_labels, name='answer_head')
        self.loss_fcts = {'l2': keras.losses.Huber(delta=1.0, name='huber_loss'), 'visn_ce': keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'ce': keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses['obj'] = {'shape': (-1,), 'num': config.num_object_labels, 'loss': 'visn_ce'}
        if config.visual_attr_loss:
            visual_losses['attr'] = {'shape': (-1,), 'num': config.num_attr_labels, 'loss': 'visn_ce'}
        if config.visual_feat_loss:
            visual_losses['feat'] = {'shape': (-1, config.visual_feat_dim), 'num': config.visual_feat_dim, 'loss': 'l2'}
        self.visual_losses = visual_losses

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        batch_size = 2
        num_visual_features = 10
        input_ids = tf.constant([[3, 5, 6], [2, 3, 4]], dtype=tf.int32)
        visual_feats = tf.random.uniform((batch_size, num_visual_features, self.config.visual_feat_dim))
        visual_pos = tf.random.uniform((batch_size, num_visual_features, 4))
        if self.config.task_obj_predict:
            obj_labels = {}
        if self.config.visual_attr_loss and self.config.task_obj_predict:
            obj_labels['attr'] = (tf.ones([batch_size, num_visual_features]), tf.ones([batch_size, num_visual_features]))
        if self.config.visual_feat_loss and self.config.task_obj_predict:
            obj_labels['feat'] = (tf.ones([batch_size, num_visual_features, self.config.visual_feat_dim]), tf.ones([batch_size, num_visual_features]))
        if self.config.visual_obj_loss and self.config.task_obj_predict:
            obj_labels['obj'] = (tf.ones([batch_size, num_visual_features]), tf.ones([batch_size, num_visual_features]))
        return {**{'input_ids': input_ids, 'visual_feats': visual_feats, 'visual_pos': visual_pos}, **({'obj_labels': obj_labels} if self.config.task_obj_predict else {})}

    def get_lm_head(self):
        return self.cls.predictions

    def get_prefix_bias_name(self):
        warnings.warn('The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.', FutureWarning)
        return self.name + '/' + self.cls.name + '/' + self.cls.predictions.name

    @unpack_inputs
    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFLxmertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids: TFModelInputType | None=None, visual_feats: tf.Tensor | None=None, visual_pos: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, visual_attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, masked_lm_labels: tf.Tensor | None=None, obj_labels: Dict[str, Tuple[tf.Tensor, tf.Tensor]] | None=None, matched_label: tf.Tensor | None=None, ans: tf.Tensor | None=None, output_attentions: bool | None=None, output_hidden_states: bool | None=None, return_dict: bool | None=None, training: bool=False) -> Tuple[tf.Tensor] | TFLxmertForPreTrainingOutput:
        """
        masked_lm_labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        obj_labels (`Dict[Str: Tuple[tf.Tensor, tf.Tensor]]`, *optional*, defaults to `None`):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            `(batch_size, num_features)` and `(batch_size, num_features, visual_feature_dim)` for each the label id and
            the label score respectively
        matched_label (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans (`tf.Tensor` of shape `(batch_size)`, *optional*, defaults to `None`):
            a one hot representation hof the correct answer *optional*

        Returns:
        """
        lxmert_output = self.lxmert(input_ids, visual_feats, visual_pos, attention_mask, visual_attention_mask, token_type_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict, training)
        lang_output, visual_output, pooled_output = (lxmert_output[0], lxmert_output[1], lxmert_output[2])
        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)
        if self.task_qa:
            answer_score = self.answer_head(pooled_output)
        else:
            answer_score = pooled_output[0][0]
        total_loss = None if masked_lm_labels is None and matched_label is None and (obj_labels is None) and (ans is None) else tf.constant(0.0)
        losses = ()
        if masked_lm_labels is not None and self.task_mask_lm:
            masked_lm_loss = self.loss_fcts['ce'](tf.reshape(masked_lm_labels, [-1]), tf.reshape(lang_prediction_scores, [-1, self.config.vocab_size]))
            total_loss += masked_lm_loss
            losses += (masked_lm_loss,)
        if matched_label is not None and self.task_matched:
            matched_loss = self.loss_fcts['ce'](tf.reshape(matched_label, [-1]), tf.reshape(cross_relationship_score, [-1, 2]))
            total_loss += matched_loss
            losses += (matched_loss,)
        if obj_labels is not None and self.task_obj_predict:
            total_visn_loss = 0.0
            visn_prediction_scores_dict = self.obj_predict_head(visual_output)
            for key, key_info in self.visual_losses.items():
                label, mask_conf = obj_labels[key]
                output_dim = key_info['num']
                loss_fct_name = key_info['loss']
                label_shape = key_info['shape']
                weight = self.visual_loss_normalizer
                visn_loss_fct = self.loss_fcts[loss_fct_name]
                visn_prediction_scores = visn_prediction_scores_dict[key]
                visn_loss = visn_loss_fct(tf.reshape(label, label_shape), tf.reshape(visn_prediction_scores, [-1, output_dim]))
                if visn_loss.ndim > 1:
                    visn_loss = tf.reduce_mean(visn_loss)
                visn_loss = tf.reduce_mean(visn_loss * tf.cast(tf.reshape(mask_conf, [-1]), visn_loss.dtype)) * weight
                total_visn_loss += visn_loss
                losses += (visn_loss,)
            total_loss += total_visn_loss
        if ans is not None and self.task_qa:
            answer_loss = self.loss_fcts['ce'](tf.reshape(ans, [-1]), tf.reshape(answer_score, [-1, self.num_qa_labels]))
            total_loss += answer_loss
            losses += (answer_loss,)
        if not return_dict:
            output = (lang_prediction_scores, cross_relationship_score, answer_score) + lxmert_output[3:]
            return (total_loss,) + output if total_loss is not None else output
        return TFLxmertForPreTrainingOutput(loss=total_loss, prediction_logits=lang_prediction_scores, cross_relationship_score=cross_relationship_score, question_answering_score=answer_score, language_hidden_states=lxmert_output.language_hidden_states, vision_hidden_states=lxmert_output.vision_hidden_states, language_attentions=lxmert_output.language_attentions, vision_attentions=lxmert_output.vision_attentions, cross_encoder_attentions=lxmert_output.cross_encoder_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'lxmert', None) is not None:
            with tf.name_scope(self.lxmert.name):
                self.lxmert.build(None)
        if getattr(self, 'cls', None) is not None:
            with tf.name_scope(self.cls.name):
                self.cls.build(None)
        if getattr(self, 'obj_predict_head', None) is not None:
            with tf.name_scope(self.obj_predict_head.name):
                self.obj_predict_head.build(None)
        if getattr(self, 'answer_head', None) is not None:
            with tf.name_scope(self.answer_head.name):
                self.answer_head.build(None)