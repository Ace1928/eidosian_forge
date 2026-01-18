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
class DataFrameIterator(BatchFromFilesMixin, Iterator):
    """Iterator capable of reading images from a directory as a dataframe."""
    allowed_class_modes = {'binary', 'categorical', 'input', 'multi_output', 'raw', 'sparse', None}

    def __init__(self, dataframe, directory=None, image_data_generator=None, x_col='filename', y_col='class', weight_col=None, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, data_format='channels_last', save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest', keep_aspect_ratio=False, dtype='float32', validate_filenames=True):
        super().set_processing_attrs(image_data_generator, target_size, color_mode, data_format, save_to_dir, save_prefix, save_format, subset, interpolation, keep_aspect_ratio)
        df = dataframe.copy()
        self.directory = directory or ''
        self.class_mode = class_mode
        self.dtype = dtype
        self._check_params(df, x_col, y_col, weight_col, classes)
        if validate_filenames:
            df = self._filter_valid_filepaths(df, x_col)
        if class_mode not in ['input', 'multi_output', 'raw', None]:
            df, classes = self._filter_classes(df, y_col, classes)
            num_classes = len(classes)
            self.class_indices = dict(zip(classes, range(len(classes))))
        if self.split:
            num_files = len(df)
            start = int(self.split[0] * num_files)
            stop = int(self.split[1] * num_files)
            df = df.iloc[start:stop, :]
        if class_mode not in ['input', 'multi_output', 'raw', None]:
            self.classes = self.get_classes(df, y_col)
        self.filenames = df[x_col].tolist()
        self._sample_weight = df[weight_col].values if weight_col else None
        if class_mode == 'multi_output':
            self._targets = [np.array(df[col].tolist()) for col in y_col]
        if class_mode == 'raw':
            self._targets = df[y_col].values
        self.samples = len(self.filenames)
        validated_string = 'validated' if validate_filenames else 'non-validated'
        if class_mode in ['input', 'multi_output', 'raw', None]:
            io_utils.print_msg(f'Found {self.samples} {validated_string} image filenames.')
        else:
            io_utils.print_msg(f'Found {self.samples} {validated_string} image filenames belonging to {num_classes} classes.')
        self._filepaths = [os.path.join(self.directory, fname) for fname in self.filenames]
        super().__init__(self.samples, batch_size, shuffle, seed)

    def _check_params(self, df, x_col, y_col, weight_col, classes):
        if self.class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'.format(self.class_mode, self.allowed_class_modes))
        if self.class_mode == 'multi_output' and (not isinstance(y_col, list)):
            raise TypeError('If class_mode="{}", y_col must be a list. Received {}.'.format(self.class_mode, type(y_col).__name__))
        if not all(df[x_col].apply(lambda x: isinstance(x, str))):
            raise TypeError(f'All values in column x_col={x_col} must be strings.')
        if self.class_mode in {'binary', 'sparse'}:
            if not all(df[y_col].apply(lambda x: isinstance(x, str))):
                raise TypeError('If class_mode="{}", y_col="{}" column values must be strings.'.format(self.class_mode, y_col))
        if self.class_mode == 'binary':
            if classes:
                classes = set(classes)
                if len(classes) != 2:
                    raise ValueError('If class_mode="binary" there must be 2 classes. {} class/es were given.'.format(len(classes)))
            elif df[y_col].nunique() != 2:
                raise ValueError('If class_mode="binary" there must be 2 classes. Found {} classes.'.format(df[y_col].nunique()))
        if self.class_mode == 'categorical':
            types = (str, list, tuple)
            if not all(df[y_col].apply(lambda x: isinstance(x, types))):
                raise TypeError('If class_mode="{}", y_col="{}" column values must be type string, list or tuple.'.format(self.class_mode, y_col))
        if classes and self.class_mode in {'input', 'multi_output', 'raw', None}:
            warnings.warn('`classes` will be ignored given the class_mode="{}"'.format(self.class_mode))
        if weight_col and (not issubclass(df[weight_col].dtype.type, np.number)):
            raise TypeError(f'Column weight_col={weight_col} must be numeric.')

    def get_classes(self, df, y_col):
        labels = []
        for label in df[y_col]:
            if isinstance(label, (list, tuple)):
                labels.append([self.class_indices[lbl] for lbl in label])
            else:
                labels.append(self.class_indices[label])
        return labels

    @staticmethod
    def _filter_classes(df, y_col, classes):
        df = df.copy()

        def remove_classes(labels, classes):
            if isinstance(labels, (list, tuple)):
                labels = [cls for cls in labels if cls in classes]
                return labels or None
            elif isinstance(labels, str):
                return labels if labels in classes else None
            else:
                raise TypeError('Expect string, list or tuple but found {} in {} column '.format(type(labels), y_col))
        if classes:
            classes = list(collections.OrderedDict.fromkeys(classes).keys())
            df[y_col] = df[y_col].apply(lambda x: remove_classes(x, classes))
        else:
            classes = set()
            for v in df[y_col]:
                if isinstance(v, (list, tuple)):
                    classes.update(v)
                else:
                    classes.add(v)
            classes = sorted(classes)
        return (df.dropna(subset=[y_col]), classes)

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

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        if self.class_mode in {'multi_output', 'raw'}:
            return self._targets
        else:
            return self.classes

    @property
    def sample_weight(self):
        return self._sample_weight