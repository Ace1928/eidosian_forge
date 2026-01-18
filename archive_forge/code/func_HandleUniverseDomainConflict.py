from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import textwrap
from googlecloudsdk.command_lib.util import check_browser
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def HandleUniverseDomainConflict(new_universe_domain, account):
    """Prompt the user to update the universe domain if there is conflict.

  If the given universe domain is different from the core/universe_domain
  property, prompt the user to update the core/universe_domain property.

  Args:
    new_universe_domain: str, The given new universe domain.
    account: str, The account name to use.
  """
    current_universe_domain = properties.VALUES.core.universe_domain.Get()
    if current_universe_domain == new_universe_domain:
        return
    message = textwrap.dedent('        WARNING: This account [{0}] is from the universe domain [{1}],\n        which does not match the current core/universe property [{2}].\n\n        Do you want to set property [core/universe_domain] to [{1}]? [Y/N]\n        ').format(account, new_universe_domain, current_universe_domain)
    should_update_universe_domain = console_io.PromptContinue(message=message)
    if should_update_universe_domain:
        properties.PersistProperty(properties.VALUES.core.universe_domain, new_universe_domain)
        log.status.Print('Updated property [core/universe_domain].')