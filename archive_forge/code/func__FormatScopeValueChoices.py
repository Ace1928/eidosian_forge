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
def _FormatScopeValueChoices(scope_value_choices):
    """Formats scope value choices for prompting and adds deprecation states."""
    choice_names, choice_mapping = ([], [])
    for scope in sorted(list(scope_value_choices.keys()), key=operator.attrgetter('flag_name')):
        for choice_resource in sorted(scope_value_choices[scope], key=operator.attrgetter('name')):
            deprecated = getattr(choice_resource, 'deprecated', None)
            if deprecated is not None:
                choice_name = '{0} ({1})'.format(choice_resource.name, deprecated.state)
            else:
                choice_name = choice_resource.name
            if len(scope_value_choices) > 1:
                if choice_name:
                    choice_name = '{0}: {1}'.format(scope.flag_name, choice_name)
                else:
                    choice_name = scope.flag_name
            choice_mapping.append((scope, choice_resource.name))
            choice_names.append(choice_name)
    return (choice_names, choice_mapping)