from the command line arguments and returns a list of URLs to be given to the
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import io
import os
import sys
import textwrap
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.ml_engine import uploads
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
from six.moves import map
from setuptools import setup, find_packages
class SysExecutableMissingError(UploadFailureError):
    """Error indicating that sys.executable was empty."""

    def __init__(self):
        super(SysExecutableMissingError, self).__init__(textwrap.dedent('        No Python executable found on path. A Python executable with setuptools\n        installed on the PYTHONPATH is required for building AI Platform training jobs.\n        '))