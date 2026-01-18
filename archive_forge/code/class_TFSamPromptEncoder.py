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
class TFSamPromptEncoder(keras.layers.Layer):

    def __init__(self, config: SamPromptEncoderConfig, shared_patch_embedding, **kwargs):
        super().__init__(**kwargs)
        self.shared_embedding = shared_patch_embedding
        self.mask_embed = TFSamMaskEmbedding(config, name='mask_embed')
        self.no_mask_embed = None
        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.input_image_size = config.image_size
        self.point_embed = []
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = None
        self.config = config

    def build(self, input_shape=None):
        self.no_mask_embed = self.add_weight(name='no_mask_embed.weight', shape=(1, self.hidden_size), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02), trainable=True)
        self.point_embed = [self.add_weight(name=f'point_embed_._{i}.weight', shape=(1, self.hidden_size), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02), trainable=True) for i in range(self.config.num_point_embeddings)]
        self.not_a_point_embed = self.add_weight(name='not_a_point_embed.weight', shape=(1, self.hidden_size), initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02), trainable=True)
        with tf.name_scope('mask_embed'):
            self.mask_embed.build((None, self.config.mask_input_channels, self.config.image_size, self.config.image_size))
        if self.built:
            return
        self.built = True
        if getattr(self, 'mask_embed', None) is not None:
            with tf.name_scope(self.mask_embed.name):
                self.mask_embed.build(None)

    def _embed_points(self, points: tf.Tensor, labels: tf.Tensor, pad: bool) -> tf.Tensor:
        """Embeds point prompts."""
        points = points + 0.5
        if pad:
            target_point_shape = (shape_list(points)[0], shape_list(points)[1], 1, shape_list(points)[-1])
            target_labels_shape = (shape_list(points)[0], shape_list(points)[1], 1)
            padding_point = tf.zeros(target_point_shape, dtype=points.dtype)
            padding_label = -tf.ones(target_labels_shape, dtype=labels.dtype)
            points = tf.concat([points, padding_point], axis=2)
            labels = tf.concat([labels, padding_label], axis=2)
        input_shape = (self.input_image_size, self.input_image_size)
        point_embedding = self.shared_embedding(points, input_shape)
        point_embedding = tf.where(labels[..., None] == -1, self.not_a_point_embed[0], point_embedding)
        point_embedding = tf.where(labels[..., None] != -10, point_embedding, tf.zeros_like(point_embedding))
        point_embedding = tf.where((labels == 0)[:, :, :, None], point_embedding + self.point_embed[0], point_embedding)
        point_embedding = tf.where((labels == 1)[:, :, :, None], point_embedding + self.point_embed[1], point_embedding)
        return point_embedding

    def _embed_boxes(self, boxes: tf.Tensor) -> tf.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5
        batch_size, nb_boxes = shape_list(boxes)[:2]
        coords = tf.reshape(boxes, (batch_size, nb_boxes, 2, 2))
        input_shape = (self.input_image_size, self.input_image_size)
        corner_embedding = self.shared_embedding(coords, input_shape)
        corner_embedding += tf.where(tf.range(shape_list(corner_embedding)[2])[None, None, :, None] == 0, self.point_embed[2][0], self.point_embed[3][0])
        return corner_embedding

    def call(self, batch_size: Optional[int], input_points: Optional[Tuple[tf.Tensor, tf.Tensor]], input_labels: tf.Tensor | None, input_boxes: tf.Tensor | None, input_masks: tf.Tensor | None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`tf.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`tf.Tensor`, *optional*):
                boxes to embed
            masks (`tf.Tensor`, *optional*):
                masks to embed
        """
        sparse_embeddings = None
        if input_points is not None:
            batch_size, point_batch_size = shape_list(input_points)[:2]
            if input_labels is None:
                raise ValueError('If points are provided, labels must also be provided.')
            point_embeddings = self._embed_points(input_points, input_labels, pad=input_boxes is None)
            sparse_embeddings = tf.zeros((batch_size, point_batch_size, 0, self.hidden_size), dtype=point_embeddings.dtype)
            sparse_embeddings = tf.concat([sparse_embeddings, point_embeddings], axis=2)
        if input_boxes is not None:
            batch_size = shape_list(input_boxes)[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = tf.concat([sparse_embeddings, box_embeddings], axis=2)
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed[0]
            dense_embeddings = tf.reshape(dense_embeddings, (1, -1, 1, 1))
            dense_embeddings = tf.tile(dense_embeddings, (batch_size, 1, self.image_embedding_size[0], self.image_embedding_size[1]))
        if sparse_embeddings is None:
            sparse_embeddings = tf.zeros((batch_size, 0, 1, self.hidden_size), dtype=dense_embeddings.dtype)
        return (sparse_embeddings, dense_embeddings)