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
def _ResolveUnderspecifiedNames(self, underspecified_names, default_scope, scope_lister, project, api_resource_registry, with_project=True):
    """Attempt to resolve scope for unresolved names.

    If unresolved_names was generated with _GetRefsAndUnderspecifiedNames
    changing them will change corresponding elements of refs list.

    Args:
      underspecified_names: list of one-items lists containing str
      default_scope: default scope for the resources
      scope_lister: callback used to list potential scopes for the resources
      project: str, id of the project
      api_resource_registry: resources Registry
      with_project: indicates whether or not project is associated. It should be
        False for flexible resource APIs

    Raises:
      UnderSpecifiedResourceError: when resource scope can't be resolved.
    """
    if not underspecified_names:
        return
    names = [n[0] for n in underspecified_names]
    if not console_io.CanPrompt():
        raise UnderSpecifiedResourceError(names, [s.flag for s in self.scopes])
    resource_scope_enum, scope_value = scope_prompter.PromptForScope(self.resource_name, names, [s.scope_enum for s in self.scopes], default_scope.scope_enum if default_scope is not None else None, scope_lister)
    if resource_scope_enum is None:
        raise UnderSpecifiedResourceError(names, [s.flag for s in self.scopes])
    resource_scope = self.scopes[resource_scope_enum]
    if with_project:
        params = {'project': project}
    else:
        params = {}
    if resource_scope.scope_enum != compute_scope.ScopeEnum.GLOBAL:
        params[resource_scope.scope_enum.param_name] = scope_value
    for name in underspecified_names:
        name[0] = api_resource_registry.Parse(name[0], params=params, collection=resource_scope.collection, enforce_collection=True)