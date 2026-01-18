import abc
import threading
from absl import logging
from tensorflow.python.util.tf_export import keras_export
def _call_on_interval(self):
    while not self.thread_stop_event.is_set():
        self.on_interval()
        self.thread_stop_event.wait(self.interval)