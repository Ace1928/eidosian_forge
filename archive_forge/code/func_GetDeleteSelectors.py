from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.run.integrations.typekits import base
def GetDeleteSelectors(self, integration_name):
    """Selectors for deleting the integration.

    Args:
      integration_name: str, name of integration.

    Returns:
      list of dict typed names.
    """
    selectors = super(RedisTypeKit, self).GetDeleteSelectors(integration_name)
    selectors.append({'type': 'vpc', 'name': '*'})
    return selectors