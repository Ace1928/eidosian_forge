import atexit
import datetime
import re
import sqlite3
import threading
from pathlib import Path
from decorator import decorator
from traitlets import (
from traitlets.config.configurable import LoggingConfigurable
from IPython.paths import locate_profile
from IPython.utils.decorators import undoc
class HistorySavingThread(threading.Thread):
    """This thread takes care of writing history to the database, so that
    the UI isn't held up while that happens.

    It waits for the HistoryManager's save_flag to be set, then writes out
    the history cache. The main thread is responsible for setting the flag when
    the cache size reaches a defined threshold."""
    daemon = True
    stop_now = False
    enabled = True

    def __init__(self, history_manager):
        super(HistorySavingThread, self).__init__(name='IPythonHistorySavingThread')
        self.history_manager = history_manager
        self.enabled = history_manager.enabled

    @only_when_enabled
    def run(self):
        atexit.register(self.stop)
        try:
            self.db = sqlite3.connect(str(self.history_manager.hist_file), **self.history_manager.connection_options)
            while True:
                self.history_manager.save_flag.wait()
                if self.stop_now:
                    self.db.close()
                    return
                self.history_manager.save_flag.clear()
                self.history_manager.writeout_cache(self.db)
        except Exception as e:
            print('The history saving thread hit an unexpected error (%s).History will not be written to the database.' % repr(e))
        finally:
            atexit.unregister(self.stop)

    def stop(self):
        """This can be called from the main thread to safely stop this thread.

        Note that it does not attempt to write out remaining history before
        exiting. That should be done by calling the HistoryManager's
        end_session method."""
        self.stop_now = True
        self.history_manager.save_flag.set()
        self.join()