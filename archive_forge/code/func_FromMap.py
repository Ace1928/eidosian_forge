from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import enum
import functools
import re
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.api_lib.compute.regions import service as regions_service
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import scope_prompter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.util import text
import six
@staticmethod
def FromMap(resource_name, scopes_map, scope_flag_prefix=None):
    """Initilize ResourceResolver instance.

    Args:
      resource_name: str, human readable name for resources eg
                     "instance group".
      scopes_map: dict, with keys should be instances of ScopeEnum, values
              should be instances of ResourceArgScope.
      scope_flag_prefix: str, prefix of flags specyfying scope.
    Returns:
      New instance of ResourceResolver.
    """
    scopes = ResourceArgScopes(flag_prefix=scope_flag_prefix)
    for scope, resource in six.iteritems(scopes_map):
        scopes.AddScope(scope, resource)
    return ResourceResolver(scopes, resource_name)