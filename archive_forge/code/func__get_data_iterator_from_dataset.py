import multiprocessing
import os
import random
import time
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def _get_data_iterator_from_dataset(dataset, dataset_type_spec):
    """Get the iterator from a dataset.

    Args:
        dataset :  A `tf.data.Dataset` object or a list/tuple of arrays.
        dataset_type_spec : the type of the dataset

    Raises:
        ValueError:
                  - If the dataset is empty.
                  - If the dataset is not a `tf.data.Dataset` object
                    or a list/tuple of arrays.
                  - If the dataset is a list/tuple of arrays and the
                    length of the list/tuple is not equal to the number

    Returns:
        iterator: An `iterator` object.
    """
    if dataset_type_spec == list:
        if len(dataset) == 0:
            raise ValueError('Received an empty list dataset. Please provide a non-empty list of arrays.')
        if _get_type_spec(dataset[0]) is np.ndarray:
            expected_shape = dataset[0].shape
            for i, element in enumerate(dataset):
                if np.array(element).shape[0] != expected_shape[0]:
                    raise ValueError(f'Received a list of NumPy arrays with different lengths. Mismatch found at index {i}, Expected shape={expected_shape} Received shape={np.array(element).shape}.Please provide a list of NumPy arrays with the same length.')
        else:
            raise ValueError(f'Expected a list of `numpy.ndarray` objects,Received: {type(dataset[0])}')
        return iter(zip(*dataset))
    elif dataset_type_spec == tuple:
        if len(dataset) == 0:
            raise ValueError('Received an empty list dataset.Please provide a non-empty tuple of arrays.')
        if _get_type_spec(dataset[0]) is np.ndarray:
            expected_shape = dataset[0].shape
            for i, element in enumerate(dataset):
                if np.array(element).shape[0] != expected_shape[0]:
                    raise ValueError(f'Received a tuple of NumPy arrays with different lengths. Mismatch found at index {i}, Expected shape={expected_shape} Received shape={np.array(element).shape}.Please provide a tuple of NumPy arrays with the same length.')
        else:
            raise ValueError(f'Expected a tuple of `numpy.ndarray` objects, Received: {type(dataset[0])}')
        return iter(zip(*dataset))
    elif dataset_type_spec == tf.data.Dataset:
        if is_batched(dataset):
            dataset = dataset.unbatch()
        return iter(dataset)
    elif dataset_type_spec == np.ndarray:
        return iter(dataset)