import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
def FormatDataTransferIdentifiers(client, transfer_identifier: str) -> str:
    """Formats a transfer config or run identifier.

  Transfer configuration/run commands should be able to support different
  formats of how the user could input the project information. This function
  will take the user input and create a uniform transfer config or
  transfer run reference that can be used for various commands.

  This function will also set the client's project id to the specified
  project id.

  Returns:
    The formatted transfer config or run.
  """
    formatted_identifier = transfer_identifier
    match = re.search('projects/([^/]+)', transfer_identifier)
    if not match:
        formatted_identifier = 'projects/' + client.GetProjectReference().projectId + '/' + transfer_identifier
    else:
        client.project_id = match.group(1)
    return formatted_identifier