import numpy as np
from tensorboard.plugins.pr_curve import metadata
def _create_tensor_summary(name, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, num_thresholds=None, display_name=None, description=None, collections=None):
    """A private helper method for generating a tensor summary.

    We use a helper method instead of having `op` directly call `raw_data_op`
    to prevent the scope of `raw_data_op` from being embedded within `op`.

    Arguments are the same as for raw_data_op.

    Returns:
      A tensor summary that collects data for PR curves.
    """
    import tensorflow.compat.v1 as tf
    summary_metadata = metadata.create_summary_metadata(display_name=display_name if display_name is not None else name, description=description or '', num_thresholds=num_thresholds)
    combined_data = tf.stack([tf.cast(true_positive_counts, tf.float32), tf.cast(false_positive_counts, tf.float32), tf.cast(true_negative_counts, tf.float32), tf.cast(false_negative_counts, tf.float32), tf.cast(precision, tf.float32), tf.cast(recall, tf.float32)])
    return tf.summary.tensor_summary(name='pr_curves', tensor=combined_data, collections=collections, summary_metadata=summary_metadata)