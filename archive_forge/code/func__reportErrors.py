import unittest as python_unittest
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
from _pydevd_bundle import pydevd_io
import traceback
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
from io import StringIO
def _reportErrors(self, errors, failures, captured_output, test_name, diff_time=''):
    error_contents = []
    for test, s in errors + failures:
        if type(s) == type((1,)):
            sio = StringIO()
            traceback.print_exception(s[0], s[1], s[2], file=sio)
            s = sio.getvalue()
        error_contents.append(s)
    sep = '\n' + self.separator1
    error_contents = sep.join(error_contents)
    if errors and (not failures):
        try:
            pydev_runfiles_xml_rpc.notifyTest('error', captured_output, error_contents, test.__pydev_pyfile__, test_name, diff_time)
        except:
            file_start = error_contents.find('File "')
            file_end = error_contents.find('", ', file_start)
            if file_start != -1 and file_end != -1:
                file = error_contents[file_start + 6:file_end]
            else:
                file = '<unable to get file>'
            pydev_runfiles_xml_rpc.notifyTest('error', captured_output, error_contents, file, test_name, diff_time)
    elif failures and (not errors):
        pydev_runfiles_xml_rpc.notifyTest('fail', captured_output, error_contents, test.__pydev_pyfile__, test_name, diff_time)
    else:
        pydev_runfiles_xml_rpc.notifyTest('error', captured_output, error_contents, test.__pydev_pyfile__, test_name, diff_time)