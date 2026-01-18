import io
import os
import re
import struct
import subprocess
import tempfile
from unittest import mock
from oslo_utils import units
from glance.common import format_inspector
from glance.tests import utils as test_utils
def _create_img(self, fmt, size, subformat=None):
    if fmt == 'vhd':
        fmt = 'vpc'
    opt = ''
    prefix = 'glance-unittest-formatinspector-'
    if subformat:
        opt = ' -o subformat=%s' % subformat
        prefix += subformat + '-'
    fn = tempfile.mktemp(prefix=prefix, suffix='.%s' % fmt)
    self._created_files.append(fn)
    subprocess.check_output('qemu-img create -f %s %s %s %i' % (fmt, opt, fn, size), shell=True)
    return fn