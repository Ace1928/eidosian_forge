from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import os
import re
import sys
import types
import uuid
import argcomplete
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import backend
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
def __ValidateCommandOrGroupInfo(self, impl_path, allow_non_existing_modules=False, exception_if_present=None):
    """Generates the information necessary to be able to load a command group.

    The group might actually be loaded now if it is the root of the SDK, or the
    information might be saved for later if it is to be lazy loaded.

    Args:
      impl_path: str, The file path to the command implementation for this
        command or group.
      allow_non_existing_modules: True to allow this module directory to not
        exist, False to raise an exception if this module does not exist.
      exception_if_present: Exception, An exception to throw if the module
        actually exists, or None.

    Raises:
      LayoutException: If the module directory does not exist and
      allow_non_existing is False.

    Returns:
      impl_path or None if the module directory does not exist and
      allow_non_existing is True.
    """
    module_root, module = os.path.split(impl_path)
    if not pkg_resources.IsImportable(module, module_root):
        if allow_non_existing_modules:
            return None
        raise command_loading.LayoutException('The given module directory does not exist: {0}'.format(impl_path))
    elif exception_if_present:
        raise exception_if_present
    return impl_path