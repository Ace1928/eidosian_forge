import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.rnn.base_rnn import RNN
Private base class for CuDNNGRU and CuDNNLSTM layers.

    Args:
      return_sequences: Boolean. Whether to return the last output
          in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
          in addition to the output.
      go_backwards: Boolean (default False).
          If True, process the input sequence backwards and return the
          reversed sequence.
      stateful: Boolean (default False). If True, the last state
          for each sample at index i in a batch will be used as initial
          state for the sample of index i in the following batch.
      time_major: Boolean (default False). If true, the inputs and outputs will
          be in shape `(timesteps, batch, ...)`, whereas in the False case, it
          will be `(batch, timesteps, ...)`.
    