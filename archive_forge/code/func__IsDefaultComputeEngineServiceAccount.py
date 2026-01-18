from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
def _IsDefaultComputeEngineServiceAccount(email, project_number):
    """Returns true if email is used as a default Compute Engine Service Account.

  Args:
    email: Service Account email.
    project_number: Project number.

  Returns:
    Returns true if the given email is a default Compue Engine Service Account.
    Returns false otherwise.
  """
    if email == '{0}@developer.gserviceaccount.com'.format(project_number):
        return True
    if email == '{0}@project.gserviceaccount.com'.format(project_number):
        return True
    return re.search('^[0-9]+-compute@developer(\\.[^.]+\\.iam)?\\.gserviceaccount\\.com', email)