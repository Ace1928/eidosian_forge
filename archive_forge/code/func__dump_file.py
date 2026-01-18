from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def _dump_file(filename, out=None):
    if out is not None:
        outf = sys.stderr if out else sys.stdout
        print('FILE:\n%s' % open(filename).read(), file=outf)