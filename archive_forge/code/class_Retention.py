import datetime
import decimal
import glob
import numbers
import os
import shutil
import string
from functools import partial
from stat import ST_DEV, ST_INO
from . import _string_parsers as string_parsers
from ._ctime_functions import get_ctime, set_ctime
from ._datetime import aware_now
class Retention:

    @staticmethod
    def retention_count(logs, number):

        def key_log(log):
            return (-os.stat(log).st_mtime, log)
        for log in sorted(logs, key=key_log)[number:]:
            os.remove(log)

    @staticmethod
    def retention_age(logs, seconds):
        t = datetime.datetime.now().timestamp()
        for log in logs:
            if os.stat(log).st_mtime <= t - seconds:
                os.remove(log)