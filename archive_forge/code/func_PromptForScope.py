from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import operator
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.util import text
def PromptForScope(resource_name, underspecified_names, scopes, default_scope, scope_lister):
    """Prompt user to specify a scope.

  Args:
    resource_name: str, human readable name of the resource.
    underspecified_names: list(str), names which lack scope context.
    scopes: list(compute_scope.ScopeEnum), scopes to query for.
    default_scope: compute_scope.ScopeEnum, force this scope to be used.
    scope_lister: func(scopes, underspecified_names)->[str->[str]], callback to
        provide possible values for each scope.
  Returns:
    tuple of chosen scope_enum and scope value.
  """
    implicit_scope = default_scope
    if len(scopes) == 1:
        implicit_scope = scopes[0]
    if implicit_scope:
        suggested_value = _GetSuggestedScopeValue(implicit_scope)
        if suggested_value is not None:
            if _PromptDidYouMeanScope(resource_name, underspecified_names, implicit_scope, suggested_value):
                return (implicit_scope, suggested_value)
    if not scope_lister:
        return (None, None)
    scope_value_choices = scope_lister(sorted(scopes, key=operator.attrgetter('name')), underspecified_names)
    choice_names, choice_mapping = _FormatScopeValueChoices(scope_value_choices)
    if len(choice_mapping) == 1:
        suggested_resource_scope_enum = choice_mapping[0][0]
        suggested_resource = choice_mapping[0][1]
        _PromptSuggestedScopeChoice(resource_name, underspecified_names, suggested_resource_scope_enum, suggested_resource)
        return (suggested_resource_scope_enum, suggested_resource)
    resource_scope_enum, scope_value = _PromptWithScopeChoices(resource_name, underspecified_names, scope_value_choices, choice_names, choice_mapping)
    return (resource_scope_enum, scope_value)