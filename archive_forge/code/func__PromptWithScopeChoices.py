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
def _PromptWithScopeChoices(resource_name, underspecified_names, scope_value_choices, choice_names, choice_mapping):
    """Queries user to choose scope and its value."""
    title = 'For the following {0}:\n {1}\n'.format(text.Pluralize(len(underspecified_names), resource_name), '\n '.join(('- [{0}]'.format(n) for n in sorted(underspecified_names))))
    flags = ' or '.join(sorted([s.prefix + s.flag_name for s in scope_value_choices.keys()]))
    idx = console_io.PromptChoice(options=choice_names, message='{0}choose {1}:'.format(title, flags))
    if idx is None:
        return (None, None)
    else:
        return choice_mapping[idx]