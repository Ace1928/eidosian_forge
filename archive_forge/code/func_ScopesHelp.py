from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import six
def ScopesHelp():
    """Returns the command help text markdown for scopes.

  Returns:
    The command help text markdown with scope intro text, aliases, and optional
    notes and/or warnings.
  """
    aliases = []
    for alias, value in sorted(six.iteritems(SCOPES)):
        if alias in DEPRECATED_SCOPE_ALIASES:
            alias = '{} (deprecated)'.format(alias)
        aliases.append('{0} | {1}'.format(alias, value[0]))
        for item in value[1:]:
            aliases.append('| ' + item)
    return 'SCOPE can be either the full URI of the scope or an alias. *Default* scopes are\nassigned to all instances. Available aliases are:\n\nAlias | URI\n--- | ---\n{aliases}\n\n{scope_deprecation_msg}\n'.format(aliases='\n'.join(aliases), scope_deprecation_msg=DEPRECATED_SCOPES_MESSAGES)