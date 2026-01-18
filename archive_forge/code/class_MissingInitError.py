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
class MissingInitError(UploadFailureError):
    """Error indicating that the package to build had no __init__.py file."""

    def __init__(self, package_dir):
        super(MissingInitError, self).__init__(textwrap.dedent('        [{}] is not a valid Python package because it does not contain an         `__init__.py` file. Please create one and try again. Also, please         ensure that --package-path refers to a local directory.\n        ').format(package_dir))