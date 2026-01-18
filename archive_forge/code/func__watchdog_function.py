import faulthandler
import os
import sys
import threading
import time
from absl import logging
def _watchdog_function(self):
    """The watchdog thread."""
    logging.info('Starting watchdog thread with timeout %r', self._timeout)
    while not self._stopped:
        time.sleep(self._timeout / 10.0)
        current_time = time.time()
        if current_time - self._last_activity_time >= self._timeout:
            logging.warning('No activity for ClusterCoordinator for %r seconds. Dumping stack traces.', self._timeout)
            if self._on_triggered:
                self._on_triggered()
            faulthandler.dump_traceback(file=self._traceback_file)
            self._traceback_file.write('==== End of stack traces ====\n')
            self._last_activity_time = current_time