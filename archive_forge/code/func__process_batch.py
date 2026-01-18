import collections
import tensorflow.compat.v2 as tf
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.layers.rnn import gru_lstm_utils
from keras.src.layers.rnn.base_cudnn_rnn import _CuDNNRNN
from tensorflow.python.util.tf_export import keras_export
def _process_batch(self, inputs, initial_state):
    if not self.time_major:
        inputs = tf.transpose(inputs, perm=(1, 0, 2))
    input_h = initial_state[0]
    input_h = tf.expand_dims(input_h, axis=0)
    params = gru_lstm_utils.canonical_to_params(weights=[self.kernel[:, self.units:self.units * 2], self.kernel[:, :self.units], self.kernel[:, self.units * 2:], self.recurrent_kernel[:, self.units:self.units * 2], self.recurrent_kernel[:, :self.units], self.recurrent_kernel[:, self.units * 2:]], biases=[self.bias[self.units:self.units * 2], self.bias[:self.units], self.bias[self.units * 2:self.units * 3], self.bias[self.units * 4:self.units * 5], self.bias[self.units * 3:self.units * 4], self.bias[self.units * 5:]], shape=self._vector_shape)
    args = {'input': inputs, 'input_h': input_h, 'input_c': 0, 'params': params, 'is_training': True, 'rnn_mode': 'gru'}
    outputs, h, _, _, _ = tf.raw_ops.CudnnRNNV2(**args)
    if self.stateful or self.return_state:
        h = h[0]
    if self.return_sequences:
        if self.time_major:
            output = outputs
        else:
            output = tf.transpose(outputs, perm=(1, 0, 2))
    else:
        output = outputs[-1]
    return (output, [h])