from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def CheckOverridesExist(self, existing_threat_prevention_profile_object, update_mask, override):
    """Checks if override exists in the current threat prevention object.

    Args:
      existing_threat_prevention_profile_object: Existing Threat Prevention
        Profile JSON object.
      update_mask: String Arg specifying type of override which needs update.
      override: The override object provided from the command line.

    Returns:
      A bool specifying if the override exists and index of the override in
      existing_threat_prevention_profile_object if the override exists or None
      is returned.
    """
    update_field = ''
    if update_mask == 'severityOverrides':
        update_field = 'severity'
    elif update_mask == 'threatOverrides':
        update_field = 'threatId'
    for i in range(0, len(existing_threat_prevention_profile_object.get(update_mask))):
        if existing_threat_prevention_profile_object.get(update_mask)[i].get(update_field) == override.get(update_field):
            return (True, i)
    return (False, None)