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
Creates a queue filled from a numpy array or pandas `DataFrame`.

    Returns a queue filled with the rows of the given (`OrderedDict` of) array
    or `DataFrame`. In the case of a pandas `DataFrame`, the first enqueued
    `Tensor` corresponds to the index of the `DataFrame`. For (`OrderedDict` of)
    numpy arrays, the first enqueued `Tensor` contains the row number.

  Args:
    data: a numpy `ndarray`, `OrderedDict` of numpy arrays, or a generator
      yielding `dict`s of numpy arrays or pandas `DataFrame` that will be read
      into the queue.
    capacity: the capacity of the queue.
    shuffle: whether or not to shuffle the rows of the array.
    min_after_dequeue: minimum number of elements that can remain in the queue
      after a dequeue operation. Only used when `shuffle` is true. If not set,
      defaults to `capacity` / 4.
    num_threads: number of threads used for reading and enqueueing.
    seed: used to seed shuffling and reader starting points.
    name: a scope name identifying the data.
    enqueue_size: the number of rows to enqueue per step.
    num_epochs: limit enqueuing to a specified number of epochs, if provided.
    pad_value: default value for dynamic padding of data samples, if provided.

  Returns:
    A queue filled with the rows of the given (`OrderedDict` of) array or
      `DataFrame`.

  Raises:
    TypeError: `data` is not a Pandas `DataFrame`, an `OrderedDict` of numpy
      arrays, a numpy `ndarray`, or a generator producing these.
    NotImplementedError: padding and shuffling data at the same time.
    NotImplementedError: padding usage with non generator data type.
  