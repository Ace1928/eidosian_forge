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
def _test_format(self, format_name, subformat=None):
    for image_size in (512, 513, 2057, 7):
        self._test_format_at_image_size(format_name, image_size * units.Mi, subformat=subformat)