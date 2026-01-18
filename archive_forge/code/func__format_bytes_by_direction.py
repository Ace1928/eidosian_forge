import codecs
import io
import os
import sys
import warnings
from ..lazy_import import lazy_import
import time
from breezy import (
from .. import config, osutils, trace
from . import NullProgressView, UIFactory
def _format_bytes_by_direction(self):
    if self._first_byte_time is None:
        bps = 0.0
    else:
        transfer_time = time.time() - self._first_byte_time
        if transfer_time < 0.001:
            transfer_time = 0.001
        bps = self._total_byte_count / transfer_time
    msg = 'Transferred: %.0fkB (%.1fkB/s r:%.0fkB w:%.0fkB' % (self._total_byte_count / 1000.0, bps / 1000.0, self._bytes_by_direction['read'] / 1000.0, self._bytes_by_direction['write'] / 1000.0)
    if self._bytes_by_direction['unknown'] > 0:
        msg += ' u:%.0fkB)' % (self._bytes_by_direction['unknown'] / 1000.0)
    else:
        msg += ')'
    return msg