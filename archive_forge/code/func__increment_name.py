from deprecated.sphinx import deprecated
import contextlib
import logging
import time
import sys
def _increment_name(name, increment=1):
    new_name = '{}_{}'.format(name, increment)
    if not _tasklogger_exists(_get_logger(new_name)):
        return new_name
    else:
        return _increment_name(name, increment=increment + 1)