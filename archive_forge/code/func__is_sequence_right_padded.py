import tensorflow as tf
import tree
from keras.src.utils.nest import pack_sequence_as
def _is_sequence_right_padded(mask):
    """Check the mask tensor and see if it right padded.

    For cuDNN kernel, it uses the sequence length param to skip the tailing
    timestep. If the data is left padded, or not a strict right padding (has
    masked value in the middle of the sequence), then cuDNN kernel won't be work
    properly in those cases.

    Left padded data: [[False, False, True, True, True]].
    Right padded data: [[True, True, True, False, False]].
    Mixture of mask/unmasked data: [[True, False, True, False, False]].

    Note that for the mixed data example above, the actually data RNN should see
    are those 2 Trues (index 0 and 2), the index 1 False should be ignored and
    not pollute the internal states.

    Args:
      mask: the Boolean tensor with shape [batch, timestep]

    Returns:
      boolean scalar tensor, whether the mask is strictly right padded.
    """
    max_seq_length = tf.shape(mask)[1]
    count_of_true = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    right_padded_mask = tf.sequence_mask(count_of_true, maxlen=max_seq_length)
    return tf.reduce_all(tf.equal(tf.cast(mask, dtype='bool'), tf.cast(right_padded_mask, dtype='bool')))