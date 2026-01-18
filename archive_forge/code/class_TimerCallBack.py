import timeit
import numpy as np
from keras.src import callbacks
from keras.src.benchmarks import distribution_util
class TimerCallBack(callbacks.Callback):
    """Callback for logging time in each epoch or batch."""

    def __init__(self):
        self.times = []
        self.timer = timeit.default_timer
        self.startup_time = timeit.default_timer()
        self.recorded_startup = False

    def on_epoch_begin(self, e, logs):
        self.epoch_start_time = self.timer()

    def on_epoch_end(self, e, logs):
        self.times.append(self.timer() - self.epoch_start_time)

    def on_batch_end(self, e, logs):
        if not self.recorded_startup:
            self.startup_time = self.timer() - self.startup_time
            self.recorded_startup = True