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
def _GetResourceScopeParam(self, resource_scope, scope_value, project, api_resource_registry, with_project=True):
    """Gets the resource scope parameters."""
    if scope_value is not None:
        if resource_scope.scope_enum == compute_scope.ScopeEnum.GLOBAL:
            return None
        else:
            collection = compute_scope.ScopeEnum.CollectionForScope(resource_scope.scope_enum)
            if with_project:
                return api_resource_registry.Parse(scope_value, params={'project': project}, collection=collection).Name()
            else:
                return api_resource_registry.Parse(scope_value, params={}, collection=collection).Name()
    elif resource_scope and resource_scope.scope_enum != compute_scope.ScopeEnum.GLOBAL:
        return resource_scope.scope_enum.property_func