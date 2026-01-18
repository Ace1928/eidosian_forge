from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
def IsCompatible(self, raise_exception=False):
    """Ensure that the Python version we are using is compatible.

    This will print an error message if not compatible.

    Compatible versions are 3.6+.
    We don't guarantee support for 3.6 so we want to warn about it.

    Args:
      raise_exception: bool, True to raise an exception rather than printing
        the error and exiting.

    Raises:
      Error: If not compatible and raise_exception is True.

    Returns:
      bool, True if the version is valid, False otherwise.
    """
    error = None
    allow_py2 = encoding.GetEncodedValue(os.environ, 'CLOUDSDK_ALLOW_PY2', 'False').lower() == 'true'
    py2_error = False
    if not self.version:
        error = 'ERROR: Your current version of Python is not compatible with the Google Cloud CLI. {0}{1}\n'.format(self.SupportedVersionMessage(), self.InstallMacPythonMessage())
    elif self.version[0] < 3:
        error = 'ERROR: Python 2 is not compatible with the Google Cloud CLI. {0}{1}\n'.format(self.SupportedVersionMessage(), self.InstallMacPythonMessage())
        py2_error = True
    elif self.version < PythonVersion.MIN_REQUIRED_PY3_VERSION:
        error = 'ERROR: Python {0}.{1} is not compatible with the Google Cloud CLI. {2}{3}\n'.format(self.version[0], self.version[1], self.SupportedVersionMessage(), self.InstallMacPythonMessage())
    if error and allow_py2 and py2_error:
        sys.stderr.write(error)
        sys.stderr.write(PythonVersion.ENV_VAR_MESSAGE)
        return True
    elif error:
        if raise_exception:
            raise Error(error)
        sys.stderr.write(error)
        sys.stderr.write(PythonVersion.ENV_VAR_MESSAGE)
        return False
    if self.version < self.MIN_SUPPORTED_PY3_VERSION:
        sys.stderr.write('WARNING:  Python 3.{0}.x is no longer officially supported by the Google Cloud CLI\nand may not function correctly. {1}{2}'.format(self.version[1], self.SupportedVersionMessage(), self.InstallMacPythonMessage()))
        sys.stderr.write('\n' + PythonVersion.ENV_VAR_MESSAGE)
    elif PythonVersion.UPCOMING_PY3_MIN_SUPPORTED_VERSION and self.version <= PythonVersion.UPCOMING_PY3_MIN_SUPPORTED_VERSION:
        sys.stderr.write('WARNING:  Python 3.{0}-3.{1} will be deprecated on {2}. {3}{4}'.format(PythonVersion.MIN_SUNSET_PY3_VERSION[1], PythonVersion.MAX_SUNSET_PY3_VERSION[1], PythonVersion.UPCOMING_PY3_DEPRECATION_DATE, self.UpcomingSupportedVersionMessage(), self.InstallMacPythonMessage()))
        sys.stderr.write('\n' + PythonVersion.ENV_VAR_MESSAGE)
    return True