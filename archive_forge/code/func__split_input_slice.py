import logging
import numpy as np
from .base import mx_real_t
from . import ndarray as nd
from .context import cpu
from .io import DataDesc
def _split_input_slice(batch_size, work_load_list):
    """Get input slice from the input shape.

    Parameters
    ----------
    batch_size : int
        The number of samples in a mini-batch.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as `ctx`.

    Returns
    -------
    slices : list of slice
        The split slices to get a specific slice.

    Raises
    ------
    ValueError
        In case of too many splits, leading to some empty slices.
    """
    total_work_load = sum(work_load_list)
    batch_num_list = [round(work_load * batch_size / total_work_load) for work_load in work_load_list]
    batch_num_sum = sum(batch_num_list)
    if batch_num_sum < batch_size:
        batch_num_list[-1] += batch_size - batch_num_sum
    slices = []
    end = 0
    for batch_num in batch_num_list:
        begin = int(min((end, batch_size)))
        end = int(min((begin + batch_num, batch_size)))
        if begin >= end:
            raise ValueError('Too many slices. Some splits are empty.')
        slices.append(slice(begin, end))
    return slices