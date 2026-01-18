from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import random
import types as tp
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator.inputs.queues import feeding_queue_runner as fqr
class _GeneratorFeedFn(object):
    """Creates feed dictionaries from `Generator` of `dicts` of numpy arrays."""

    def __init__(self, placeholders, generator, batch_size, random_start=False, seed=None, num_epochs=None, pad_value=None):
        first_sample = next(generator())
        if len(placeholders) != len(first_sample):
            raise ValueError('Expected {} placeholders; got {}.'.format(len(first_sample), len(placeholders)))
        self._keys = sorted(list(first_sample.keys()))
        self._col_placeholders = placeholders
        self._generator_function = generator
        self._iterator = generator()
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._epoch = 0
        self._pad_value = pad_value
        random.seed(seed)

    def __call__(self):
        if self._num_epochs and self._epoch >= self._num_epochs:
            raise tf.errors.OutOfRangeError(None, None, 'Already emitted %s epochs.' % self._epoch)
        list_dict = {}
        list_dict_size = 0
        while list_dict_size < self._batch_size:
            try:
                data_row = next(self._iterator)
            except StopIteration:
                self._epoch += 1
                self._iterator = self._generator_function()
                data_row = next(self._iterator)
            for index, key in enumerate(self._keys):
                if key not in data_row.keys():
                    raise KeyError('key mismatch between dicts emitted by GenFun Expected {} keys; got {}'.format(self._keys, data_row.keys()))
                list_dict.setdefault(self._col_placeholders[index], list()).append(data_row[key])
                list_dict_size += 1
        if self._pad_value is not None:
            feed_dict = {key: np.asarray(_pad_if_needed(item, self._pad_value)) for key, item in list(list_dict.items())}
        else:
            feed_dict = {key: np.asarray(item) for key, item in list(list_dict.items())}
        return feed_dict