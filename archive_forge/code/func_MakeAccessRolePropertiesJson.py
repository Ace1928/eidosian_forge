from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def MakeAccessRolePropertiesJson(iam_role_id: str) -> str:
    """Returns properties for a connection with IAM role id.

  Args:
    iam_role_id: IAM role id.

  Returns:
    JSON string with properties to create a connection with IAM role id.
  """
    return '{"accessRole": {"iamRoleId": "%s"}}' % iam_role_id