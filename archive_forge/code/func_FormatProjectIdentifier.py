import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
def FormatProjectIdentifier(client, project_id: str) -> str:
    """Formats a project identifier.

  If the user specifies a project with "projects/${PROJECT_ID}", isolate the
  project id and return it.

  This function will also set the client's project id to the specified
  project id.

  Returns:
    The project is.
  """
    formatted_identifier = project_id
    match = re.search('projects/([^/]+)', project_id)
    if match:
        formatted_identifier = match.group(1)
        client.project_id = formatted_identifier
    return formatted_identifier