import logging
import threading
import time
import numpy as np
def _measure_and_tune(self):
    self.spe_measurement_count += 1
    cur_iteration = self.optimizer.iterations.numpy()
    cur_time_secs = time.time()
    recent_gsps = (cur_iteration - self.spe_last_logged['iteration']) / (cur_time_secs - self.spe_last_logged['time_secs'])
    self.rgsps.append(recent_gsps)
    if len(self.rgsps) > self.change_spe_interval:
        self.rgsps.pop(0)
    if cur_iteration == 0:
        self.start_time = cur_time_secs
        return
    self.spe_last_logged['iteration'] = cur_iteration
    self.spe_last_logged['time_secs'] = cur_time_secs
    try:
        if self._should_tune():
            self._tune()
    except RuntimeError:
        logging.exception('Steps per execution autotuner failed to run.')
        return