from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes as compute_base
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import scope_prompter
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def ExpandScopeAliases(scopes):
    """Replace known aliases in the list of scopes provided by the user."""
    scopes = scopes or []
    expanded_scopes = []
    for scope in scopes:
        if scope in SCOPE_ALIASES:
            expanded_scopes += SCOPE_ALIASES[scope]
        else:
            expanded_scopes.append(scope)
    return sorted(expanded_scopes)