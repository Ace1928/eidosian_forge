from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import resource_cache
from googlecloudsdk.core.resource import resource_property
def GetDestFromParam(param, prefix=None):
    """Returns a conventional dest name given param name with optional prefix."""
    name = param.replace('-', '_').strip('_')
    if prefix:
        name = prefix + '_' + name
    return resource_property.ConvertToSnakeCase(re.sub('s?I[Dd]$', '', name)).strip('_')