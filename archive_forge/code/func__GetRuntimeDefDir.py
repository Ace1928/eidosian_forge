from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from gae_ext_runtime import ext_runtime
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def _GetRuntimeDefDir():
    runtime_root = properties.VALUES.app.runtime_root.Get()
    if runtime_root:
        return runtime_root
    raise NoRuntimeRootError('Unable to determine the root directory where GAE runtimes are stored.  Please define the CLOUDSDK_APP_RUNTIME_ROOT environmnent variable.')