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
def GetScopeNameWithProjectNumber(name):
    """Rebuilds scope name with project number instead of ID."""
    delimiter = '/'
    tokens = name.split(delimiter)
    if len(tokens) != 6 or tokens[0] != 'projects':
        raise exceptions.Error('{} is not a valid Scope resource name'.format(name))
    project_id = tokens[1]
    project_number = project_util.GetProjectNumber(project_id)
    tokens[1] = six.text_type(project_number)
    return delimiter.join(tokens)