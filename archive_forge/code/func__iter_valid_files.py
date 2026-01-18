import collections
import multiprocessing
import os
import threading
import warnings
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from keras.src.utils import image_utils
from keras.src.utils import io_utils
from keras.src.utils.module_utils import scipy
def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension.

    Args:
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean, follow symbolic links to subdirectories.
    Yields:
        Tuple of (root, filename) with extension in `white_list_formats`.
    """

    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda x: x[0])
    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            if fname.lower().endswith('.tiff'):
                warnings.warn('Using ".tiff" files with multiple bands will cause distortion. Please verify your output.')
            if fname.lower().endswith(white_list_formats):
                yield (root, fname)