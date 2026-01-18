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
class ResourceArgScopes(object):
    """Represents chosen set of scopes."""

    def __init__(self, flag_prefix):
        self.flag_prefix = flag_prefix
        self.scopes = {}

    def AddScope(self, scope, collection):
        self.scopes[scope] = ResourceArgScope(scope, self.flag_prefix, collection)

    def SpecifiedByArgs(self, args):
        """Given argparse args return selected scope and its value."""
        for resource_scope in six.itervalues(self.scopes):
            scope_value = getattr(args, resource_scope.flag_name, None)
            if scope_value is not None:
                return (resource_scope, scope_value)
        return (None, None)

    def SpecifiedByValue(self, value):
        """Given resource value return selected scope and its value."""
        if re.match(_GLOBAL_RELATIVE_PATH_REGEX, value):
            return (self.scopes[compute_scope.ScopeEnum.GLOBAL], 'global')
        elif re.match(_REGIONAL_RELATIVE_PATH_REGEX, value):
            return (self.scopes[compute_scope.ScopeEnum.REGION], 'region')
        elif re.match(_ZONAL_RELATIVE_PATH_REGEX, value):
            return (self.scopes[compute_scope.ScopeEnum.ZONE], 'zone')
        return (None, None)

    def GetImplicitScope(self, default_scope=None):
        """See if there is no ambiguity even if scope is not known from args."""
        if len(self.scopes) == 1:
            return next(six.itervalues(self.scopes))
        return default_scope

    def __iter__(self):
        return iter(six.itervalues(self.scopes))

    def __contains__(self, scope):
        return scope in self.scopes

    def __getitem__(self, scope):
        return self.scopes[scope]

    def __len__(self):
        return len(self.scopes)