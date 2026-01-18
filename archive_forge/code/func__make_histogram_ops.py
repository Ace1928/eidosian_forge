import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.training import saver
def _make_histogram_ops(self, model):
    """Defines histogram ops when histogram_freq > 0."""
    if self.histogram_freq and self.merged is None:
        for layer in self.model.layers:
            for weight in layer.weights:
                mapped_weight_name = weight.name.replace(':', '_')
                tf_summary.histogram(mapped_weight_name, weight)
                if self.write_images:
                    w_img = array_ops.squeeze(weight)
                    shape = K.int_shape(w_img)
                    if len(shape) == 2:
                        if shape[0] > shape[1]:
                            w_img = array_ops.transpose(w_img)
                            shape = K.int_shape(w_img)
                        w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
                    elif len(shape) == 3:
                        if K.image_data_format() == 'channels_last':
                            w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
                            shape = K.int_shape(w_img)
                        w_img = array_ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])
                    elif len(shape) == 1:
                        w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
                    else:
                        continue
                    shape = K.int_shape(w_img)
                    assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                    tf_summary.image(mapped_weight_name, w_img)
            if self.write_grads:
                for weight in layer.trainable_weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    grads = model.optimizer.get_gradients(model.total_loss, weight)

                    def is_indexed_slices(grad):
                        return type(grad).__name__ == 'IndexedSlices'
                    grads = [grad.values if is_indexed_slices(grad) else grad for grad in grads]
                    tf_summary.histogram('{}_grad'.format(mapped_weight_name), grads)
            if hasattr(layer, 'output'):
                if isinstance(layer.output, list):
                    for i, output in enumerate(layer.output):
                        tf_summary.histogram('{}_out_{}'.format(layer.name, i), output)
                else:
                    tf_summary.histogram('{}_out'.format(layer.name), layer.output)