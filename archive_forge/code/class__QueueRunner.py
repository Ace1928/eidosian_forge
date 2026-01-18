from abc import ABCMeta, abstractmethod
import logging
import numpy as np
import threading
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.framework import try_import_tf
from typing import Dict, List
from ray.rllib.utils.typing import TensorType, SampleBatchType
class _QueueRunner(threading.Thread):
    """Thread that feeds a TF queue from a InputReader."""

    def __init__(self, input_reader: InputReader, queue: 'tf1.FIFOQueue', keys: List[str], dtypes: 'tf.dtypes.DType'):
        threading.Thread.__init__(self)
        self.sess = tf1.get_default_session()
        self.daemon = True
        self.input_reader = input_reader
        self.keys = keys
        self.queue = queue
        self.placeholders = [tf1.placeholder(dtype) for dtype in dtypes]
        self.enqueue_op = queue.enqueue(dict(zip(keys, self.placeholders)))

    def enqueue(self, batch: SampleBatchType):
        data = {self.placeholders[i]: batch[key] for i, key in enumerate(self.keys)}
        self.sess.run(self.enqueue_op, feed_dict=data)

    def run(self):
        while True:
            try:
                batch = self.input_reader.next()
                self.enqueue(batch)
            except Exception:
                logger.exception('Error reading from input')