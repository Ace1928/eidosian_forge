from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.resource_settings import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetResourceTypeFromString(setting):
    """Returns the resource type from the setting path.

  A setting path should start with following syntax:
  [organizations|folders|projects]/{resource_id}/settings/{setting_name}/value

  Args:
    setting: A String that contains the setting path
  """
    if setting.startswith('organizations/'):
        resource_type = ORGANIZATION
    elif setting.startswith('folders/'):
        resource_type = FOLDER
    elif setting.startswith('projects/'):
        resource_type = PROJECT
    else:
        resource_type = 'invalid'
    return resource_type