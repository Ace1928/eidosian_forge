from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import iam as kms_iam
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.api_lib.storage import storage_api
def GetOrCreate(project_ref):
    """Gets (or creates) the P4SA for Private CA in the given project.

  If the P4SA does not exist for this project, it will be created. Otherwise,
  the email address of the existing P4SA will be returned.

  Args:
    project_ref: resources.Resource reference to the project for the P4SA.

  Returns:
    Email address of the Private CA P4SA for the given project.
  """
    service_name = privateca_base.GetServiceName()
    response = serviceusage.GenerateServiceIdentity(project_ref.Name(), service_name)
    return response['email']