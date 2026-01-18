from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import sys
from fire import completion
from fire import custom_descriptions
from fire import decorators
from fire import docstrings
from fire import formatting
from fire import inspectutils
from fire import value_types
def HelpText(component, trace=None, verbose=False):
    """Gets the help string for the current component, suitable for a help screen.

  Args:
    component: The component to construct the help string for.
    trace: The Fire trace of the command so far. The command executed so far
      can be extracted from this trace.
    verbose: Whether to include private members in the help screen.

  Returns:
    The full help screen as a string.
  """
    info = inspectutils.Info(component)
    actions_grouped_by_kind = _GetActionsGroupedByKind(component, verbose=verbose)
    spec = inspectutils.GetFullArgSpec(component)
    metadata = decorators.GetMetadata(component)
    name_section = _NameSection(component, info, trace=trace, verbose=verbose)
    synopsis_section = _SynopsisSection(component, actions_grouped_by_kind, spec, metadata, trace=trace)
    description_section = _DescriptionSection(component, info)
    if callable(component):
        args_and_flags_sections, notes_sections = _ArgsAndFlagsSections(info, spec, metadata)
    else:
        args_and_flags_sections = []
        notes_sections = []
    usage_details_sections = _UsageDetailsSections(component, actions_grouped_by_kind)
    sections = [name_section, synopsis_section, description_section] + args_and_flags_sections + usage_details_sections + notes_sections
    return '\n\n'.join((_CreateOutputSection(*section) for section in sections if section is not None))