import logging
import threading
import time
import numpy as np
def _should_tune(self):
    epoch_boundary = False
    if self.rgsps[-1] == 0:
        epoch_boundary = True
    return self.spe_measurement_count % self.change_spe_interval == 0 and self.rgsps and (not epoch_boundary)