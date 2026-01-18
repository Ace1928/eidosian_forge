from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import runtime_registry
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import exceptions as core_exceptions
def _GetGcloudignoreRegistry():
    return runtime_registry.Registry(_GCLOUDIGNORE_REGISTRY, default=False)