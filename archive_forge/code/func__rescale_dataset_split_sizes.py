import multiprocessing
import os
import random
import time
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def _rescale_dataset_split_sizes(left_size, right_size, total_length):
    """Rescale the dataset split sizes.

    We want to ensure that the sum of
    the split sizes is equal to the total length of the dataset.

    Args:
        left_size : The size of the left dataset split.
        right_size : The size of the right dataset split.
        total_length : The total length of the dataset.

    Raises:
        TypeError: - If `left_size` or `right_size` is not an integer or float.
        ValueError: - If `left_size` or `right_size` is negative or greater
                      than 1 or greater than `total_length`.

    Returns:
        tuple: A tuple of rescaled left_size and right_size
    """
    left_size_type = type(left_size)
    right_size_type = type(right_size)
    if (left_size is not None and left_size_type not in [int, float]) and (right_size is not None and right_size_type not in [int, float]):
        raise TypeError(f'Invalid `left_size` and `right_size` Types. Expected: integer or float or None, Received: type(left_size)={left_size_type} and type(right_size)={right_size_type}')
    if left_size is not None and left_size_type not in [int, float]:
        raise TypeError(f'Invalid `left_size` Type. Expected: int or float or None, Received: type(left_size)={left_size_type}.  ')
    if right_size is not None and right_size_type not in [int, float]:
        raise TypeError(f'Invalid `right_size` Type. Expected: int or float or None,Received: type(right_size)={right_size_type}.')
    if left_size == 0 and right_size == 0:
        raise ValueError('Both `left_size` and `right_size` are zero. At least one of the split sizes must be non-zero.')
    if left_size_type == int and (left_size <= 0 or left_size >= total_length) or (left_size_type == float and (left_size <= 0 or left_size >= 1)):
        raise ValueError(f'`left_size` should be either a positive integer smaller than {total_length}, or a float within the range `[0, 1]`. Received: left_size={left_size}')
    if right_size_type == int and (right_size <= 0 or right_size >= total_length) or (right_size_type == float and (right_size <= 0 or right_size >= 1)):
        raise ValueError(f'`right_size` should be either a positive integer and smaller than {total_length} or a float within the range `[0, 1]`. Received: right_size={right_size}')
    if right_size_type == left_size_type == float and right_size + left_size > 1:
        raise ValueError('The sum of `left_size` and `right_size` is greater than 1. It must be less than or equal to 1.')
    if left_size_type == float:
        left_size = round(left_size * total_length)
    elif left_size_type == int:
        left_size = float(left_size)
    if right_size_type == float:
        right_size = round(right_size * total_length)
    elif right_size_type == int:
        right_size = float(right_size)
    if left_size is None:
        left_size = total_length - right_size
    elif right_size is None:
        right_size = total_length - left_size
    if left_size + right_size > total_length:
        raise ValueError(f'The sum of `left_size` and `right_size` should be smaller than the {{total_length}}. Received: left_size + right_size = {left_size + right_size}and total_length = {total_length}')
    for split, side in [(left_size, 'left'), (right_size, 'right')]:
        if split == 0:
            raise ValueError(f'With `dataset` of length={total_length}, `left_size`={left_size} and `right_size`={right_size}.Resulting {side} side dataset split will be empty. Adjust any of the aforementioned parameters')
    left_size, right_size = (int(left_size), int(right_size))
    return (left_size, right_size)