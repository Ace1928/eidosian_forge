import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.command_lib.scc.manage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
def ParseJSONFile(file):
    """Converts the contents of a JSON file into a string."""
    if file is not None:
        try:
            config = json.loads(file)
            return json.dumps(config)
        except json.JSONDecodeError as e:
            raise errors.InvalidConfigValueFileError('Error parsing config value file [{}]'.format(e))
    else:
        return None