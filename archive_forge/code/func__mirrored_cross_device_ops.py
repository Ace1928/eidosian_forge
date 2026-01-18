import json
import os
import tensorflow.compat.v2 as tf
def _mirrored_cross_device_ops(all_reduce_alg, num_packs):
    """Return a CrossDeviceOps based on all_reduce_alg and num_packs.

    Args:
      all_reduce_alg: a string specifying which cross device op to pick, or
        None.
      num_packs: an integer specifying number of packs for the cross device op.

    Returns:
      tf.distribute.CrossDeviceOps object or None.

    Raises:
      ValueError: if `all_reduce_alg` not in [None, "nccl",
        "hierarchical_copy"].
    """
    if all_reduce_alg is None:
        return None
    mirrored_all_reduce_options = {'nccl': tf.distribute.NcclAllReduce, 'hierarchical_copy': tf.distribute.HierarchicalCopyAllReduce}
    if all_reduce_alg not in mirrored_all_reduce_options:
        raise ValueError('When used with `mirrored`, valid values for all_reduce_alg are [`nccl`, `hierarchical_copy`].  Supplied value: {}'.format(all_reduce_alg))
    cross_device_ops_class = mirrored_all_reduce_options[all_reduce_alg]
    return cross_device_ops_class(num_packs=num_packs)