import multiprocessing
import os
import random
import time
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def _restore_dataset_from_list(dataset_as_list, dataset_type_spec, original_dataset):
    """Restore the dataset from the list of arrays."""
    if dataset_type_spec in [tuple, list]:
        return tuple((np.array(sample) for sample in zip(*dataset_as_list)))
    elif dataset_type_spec == tf.data.Dataset:
        if isinstance(original_dataset.element_spec, dict):
            restored_dataset = {}
            for d in dataset_as_list:
                for k, v in d.items():
                    if k not in restored_dataset:
                        restored_dataset[k] = [v]
                    else:
                        restored_dataset[k].append(v)
            return restored_dataset
        else:
            return tuple((np.array(sample) for sample in zip(*dataset_as_list)))
    return dataset_as_list