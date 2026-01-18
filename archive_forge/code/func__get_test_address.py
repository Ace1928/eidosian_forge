from nose.plugins.multiprocess import MultiProcessTestRunner  # @UnresolvedImport
from nose.plugins.base import Plugin  # @UnresolvedImport
import sys
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from contextlib import contextmanager
from io import StringIO
import traceback
def _get_test_address(self, test):
    try:
        if hasattr(test, 'address'):
            with self._without_user_address(test):
                address = test.address()
            address = (address[0], address[2])
        else:
            try:
                address = (test[0], test[1])
            except TypeError:
                f = test.context.__file__
                if f.endswith('.pyc'):
                    f = f[:-1]
                elif f.endswith('$py.class'):
                    f = f[:-len('$py.class')] + '.py'
                address = (f, '?')
    except:
        sys.stderr.write('PyDev: Internal pydev error getting test address. Please report at the pydev bug tracker\n')
        traceback.print_exc()
        sys.stderr.write('\n\n\n')
        address = ('?', '?')
    return address