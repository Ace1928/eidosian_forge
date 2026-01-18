from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import times
import six
@staticmethod
def GetProjectFromScopeName(name):
    """Extracts the project name from the full Scope resource name."""
    return name.split('/')[1]