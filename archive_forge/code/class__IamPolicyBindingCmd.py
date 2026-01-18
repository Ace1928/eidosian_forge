from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from typing import Optional
from absl import app
from absl import flags
import bq_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from utils import bq_id_utils
class _IamPolicyBindingCmd(_IamPolicyCmd):
    """Common superclass for AddIamPolicyBinding and RemoveIamPolicyBinding.

  Provides the flags that are common to both commands, and also inherits
  flags and logic from the _IamPolicyCmd class.
  """

    def __init__(self, name: str, fv: flags.FlagValues, verb: str):
        super(_IamPolicyBindingCmd, self).__init__(name, fv, verb)
        flags.DEFINE_string('member', None, 'The member part of the IAM policy binding. Acceptable values include "user:<email>", "group:<email>", "serviceAccount:<email>", "allAuthenticatedUsers" and "allUsers".\n\n"allUsers" is a special value that represents every user. "allAuthenticatedUsers" is a special value that represents every user that is authenticated with a Google account or a service account.\n\nExamples:\n  "user:myaccount@gmail.com"\n  "group:mygroup@example-company.com"\n  "serviceAccount:myserviceaccount@sub.example-company.com"\n  "domain:sub.example-company.com"\n  "allUsers"\n  "allAuthenticatedUsers"', flag_values=fv)
        flags.DEFINE_string('role', None, 'The role part of the IAM policy binding.\n\nExamples:\n\nA predefined (built-in) BigQuery role:\n  "roles/bigquery.dataViewer"\n\nA custom role defined in a project:\n  "projects/my-project/roles/MyCustomRole"\n\nA custom role defined in an organization:\n  "organizations/111111111111/roles/MyCustomRole"', flag_values=fv)
        flags.mark_flag_as_required('member', flag_values=fv)
        flags.mark_flag_as_required('role', flag_values=fv)