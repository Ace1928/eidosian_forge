import inspect
import logging
import os
import signal
import stat
import sys
import threading
import time
import traceback
from oslo_utils import timeutils
from oslo_reports.generators import conf as cgen
from oslo_reports.generators import process as prgen
from oslo_reports.generators import threading as tgen
from oslo_reports.generators import version as pgen
from oslo_reports import report
@classmethod
def _setup_file_watcher(cls, filepath, interval, version, service_name, log_dir):
    st = os.stat(filepath)
    if not bool(st.st_mode & stat.S_IRGRP):
        LOG.error("Guru Meditation Report does not have read permissions to '%s' file.", filepath)

    def _handler():
        mtime = time.time()
        while True:
            try:
                stat = os.stat(filepath)
                if stat.st_mtime > mtime:
                    cls.handle_signal(version, service_name, log_dir, None)
                    mtime = stat.st_mtime
            except OSError:
                msg = 'Guru Meditation Report cannot read ' + "'{0}' file".format(filepath)
                raise IOError(msg)
            finally:
                time.sleep(interval)
    th = threading.Thread(target=_handler)
    th.daemon = True
    th.start()