import io
import json
import os
import sys
from unittest import mock
import ddt
from osprofiler.cmd import shell
from osprofiler import exc
from osprofiler.tests import test
def _trace_show_cmd(self, format_=None):
    cmd = 'trace show --connection-string redis:// %s' % self.TRACE_ID
    return cmd if format_ is None else '%s --%s' % (cmd, format_)