import threading
import time
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
def _get_duration_sec(self):
    return int(time.time() - self.start_time)