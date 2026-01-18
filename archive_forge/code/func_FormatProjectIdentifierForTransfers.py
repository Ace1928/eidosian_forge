from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def FormatProjectIdentifierForTransfers(project_reference: 'bq_id_utils.ApiClientHelper.ProjectReference', location: str) -> str:
    """Formats a project identifier for data transfers.

  Data transfer API calls take in the format projects/(projectName), so because
  by default project IDs take the format (projectName), add the beginning format
  to perform data transfer commands

  Args:
    project_reference: The project id to format for data transfer commands.
    location: The location id, e.g. 'us' or 'eu'.

  Returns:
    The formatted project name for transfers.
  """
    return 'projects/' + project_reference.projectId + '/locations/' + location