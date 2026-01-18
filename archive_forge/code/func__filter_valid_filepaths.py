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
def _filter_valid_filepaths(self, df, x_col):
    """Keep only dataframe rows with valid filenames.

        Args:
            df: Pandas dataframe containing filenames in a column
            x_col: string, column in `df` that contains the filenames or
                filepaths
        Returns:
            absolute paths to image files
        """
    filepaths = df[x_col].map(lambda fname: os.path.join(self.directory, fname))
    mask = filepaths.apply(validate_filename, args=(self.white_list_formats,))
    n_invalid = (~mask).sum()
    if n_invalid:
        warnings.warn('Found {} invalid image filename(s) in x_col="{}". These filename(s) will be ignored.'.format(n_invalid, x_col))
    return df[mask]