from __future__ import annotations
import collections
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, keras, shape_list, unpack_inputs
from ...tf_utils import flatten, functional_layernorm
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
@add_start_docstrings('Segment Anything Model (SAM) for generating segmentation masks, given an input image and ', ' optional 2D location and bounding boxes.', SAM_START_DOCSTRING)
class TFSamModel(TFSamPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['prompt_encoder.shared_embedding.positional_embedding']

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.shared_image_embedding = TFSamPositionalEmbedding(config.vision_config, name='shared_image_embedding')
        self.vision_encoder = TFSamVisionEncoder(config.vision_config, name='vision_encoder')
        self.prompt_encoder = TFSamPromptEncoder(config.prompt_encoder_config, self.shared_image_embedding, name='prompt_encoder')
        self.mask_decoder = TFSamMaskDecoder(config.mask_decoder_config, name='mask_decoder')
        self.config = config

    def get_input_embeddings(self):
        return self.vision_encoder.get_input_embeddings()

    def get_image_wide_positional_embeddings(self):
        size = self.config.prompt_encoder_config.image_embedding_size
        grid = tf.ones((size, size))
        y_embed = tf.math.cumsum(grid, axis=0) - 0.5
        x_embed = tf.math.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size
        positional_embedding = self.shared_image_embedding(tf.stack([x_embed, y_embed], axis=-1))
        return tf.expand_dims(tf.transpose(positional_embedding, perm=[2, 0, 1]), axis=0)

    def get_image_embeddings(self, pixel_values, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        """
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.TFModelOutput`] instead of a plain tuple.

        """
        vision_output = self.vision_encoder(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeddings = vision_output[0]
        return image_embeddings

    def get_prompt_embeddings(self, input_points: tf.Tensor | None=None, input_labels: tf.Tensor | None=None, input_boxes: tf.Tensor | None=None, input_masks: tf.Tensor | None=None):
        """
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`tf.Tensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`tf.Tensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`tf.Tensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`tf.Tensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        prompt_output = self.prompt_encoder(input_points=input_points, input_labels=input_labels, input_boxes=input_boxes, input_masks=input_masks)
        return prompt_output

    @unpack_inputs
    @add_start_docstrings_to_model_forward(SAM_INPUTS_DOCSTRING)
    def call(self, pixel_values: TFModelInputType | None=None, input_points: tf.Tensor | None=None, input_labels: tf.Tensor | None=None, input_boxes: tf.Tensor | None=None, input_masks: tf.Tensor | None=None, image_embeddings: tf.Tensor | None=None, multimask_output: bool=True, output_attentions: bool | None=None, output_hidden_states: bool | None=None, return_dict: bool | None=None, training: bool=False, **kwargs) -> TFSamImageSegmentationOutput | Tuple[tf.Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None and image_embeddings is None:
            raise ValueError('Either pixel_values or image_embeddings must be provided.')
        if pixel_values is not None and image_embeddings is not None:
            raise ValueError('Only one of pixel_values and image_embeddings can be provided.')
        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError('The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.', ' got {}.'.format(input_points.shape))
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError('The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.', ' got {}.'.format(input_boxes.shape))
        if input_points is not None and input_boxes is not None:
            point_batch_size = shape_list(input_points)[1]
            box_batch_size = shape_list(input_boxes)[1]
            if point_batch_size != box_batch_size:
                raise ValueError('You should provide as many bounding boxes as input points per box. Got {} and {}.'.format(point_batch_size, box_batch_size))
        if pixel_values is not None:
            pixel_values = tf.ensure_shape(pixel_values, [None, self.config.vision_config.num_channels, self.config.vision_config.image_size, self.config.vision_config.image_size])
        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        batch_size = shape_list(pixel_values)[0] if pixel_values is not None else shape_list(image_embeddings)[0]
        image_positional_embeddings = tf.repeat(image_positional_embeddings, batch_size, axis=0)
        vision_attentions = None
        vision_hidden_states = None
        if pixel_values is not None:
            vision_outputs = self.vision_encoder(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True, training=training)
            image_embeddings = vision_outputs['last_hidden_state']
            if output_hidden_states:
                vision_hidden_states = vision_outputs['hidden_states']
            if output_attentions:
                vision_attentions = vision_outputs['attentions']
        if input_points is not None and input_labels is None:
            input_labels = tf.ones_like(input_points[:, :, :, 0], dtype=tf.int32)
        if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
            raise ValueError('The batch size of the image embeddings and the input points must be the same. ', 'Got {} and {} respectively.'.format(image_embeddings.shape[0], input_points.shape[0]), ' if you want to pass multiple points for the same image, make sure that you passed ', ' input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ', ' input_labels of shape (batch_size, point_batch_size, num_points_per_image)')
        sparse_embeddings, dense_embeddings = self.prompt_encoder(batch_size=shape_list(image_embeddings)[0], input_points=input_points, input_labels=input_labels, input_boxes=input_boxes, input_masks=input_masks)
        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(image_embeddings=image_embeddings, image_positional_embeddings=image_positional_embeddings, sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, output_attentions=output_attentions)
        if not return_dict:
            output = (iou_predictions, low_res_masks)
            if output_hidden_states:
                output = output + (vision_hidden_states,)
            if output_attentions:
                output = output + (vision_attentions, mask_decoder_attentions)
            return output
        return TFSamImageSegmentationOutput(iou_scores=iou_predictions, pred_masks=low_res_masks, vision_hidden_states=vision_hidden_states, vision_attentions=vision_attentions, mask_decoder_attentions=mask_decoder_attentions)

    def serving_output(self, output: TFSamImageSegmentationOutput) -> TFSamImageSegmentationOutput:
        hs = tf.convert_to_tensor(output.vision_hidden_states) if self.config.output_hidden_states else None
        attns = tf.convert_to_tensor(output.vision_attentions) if self.config.output_attentions else None
        return TFSamImageSegmentationOutput(iou_scores=output.iou_scores, pred_masks=output.pred_masks, vision_hidden_states=hs if self.config.output_hidden_states else None, vision_attentions=attns if self.config.output_attentions else None, mask_decoder_attentions=output.mask_decoder_attentions if self.config.output_attentions else None)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'shared_image_embedding', None) is not None:
            with tf.name_scope(self.shared_image_embedding.name):
                self.shared_image_embedding.build(None)
        if getattr(self, 'vision_encoder', None) is not None:
            with tf.name_scope(self.vision_encoder.name):
                self.vision_encoder.build(None)
        if getattr(self, 'prompt_encoder', None) is not None:
            with tf.name_scope(self.prompt_encoder.name):
                self.prompt_encoder.build(None)
        if getattr(self, 'mask_decoder', None) is not None:
            with tf.name_scope(self.mask_decoder.name):
                self.mask_decoder.build(None)