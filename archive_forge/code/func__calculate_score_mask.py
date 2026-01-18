from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
def _calculate_score_mask(self, scores, v_mask, use_causal_mask):
    if use_causal_mask:
        score_shape = ops.shape(scores)
        mask_shape = (1, score_shape[-2], score_shape[-1])
        ones_mask = ops.ones(shape=mask_shape, dtype='int32')
        row_index = ops.cumsum(ones_mask, axis=-2)
        col_index = ops.cumsum(ones_mask, axis=-1)
        causal_mask = ops.greater_equal(row_index, col_index)
        if v_mask is not None:
            v_mask = ops.expand_dims(v_mask, axis=-2)
            return ops.logical_and(v_mask, causal_mask)
        return causal_mask
    else:
        return v_mask