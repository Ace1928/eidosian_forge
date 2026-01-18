from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
def _IsDefaultAppEngineServiceAccount(email):
    """Returns true if email is used as a default App Engine Service Account.

  Args:
    email: Service Account email.

  Returns:
    Returns true if the given email is default App Engine Service Account.
    Returns false otherwise.
  """
    return re.search('^([\\w:.-]+)@appspot(\\.[^.]+\\.iam)?\\.gserviceaccount\\.com', email)