from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import iam as kms_iam
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.command_lib.privateca import exceptions
def CheckCreateCertificatePermissions(issuing_ca_pool_ref):
    """Ensures that the current user can issue a certificate from the given Pool.

  Args:
    issuing_ca_pool_ref: The CA pool that will create the certificate.

  Raises:
    InsufficientPermissionException: If the user is missing permissions.
  """
    client = privateca_base.GetClientInstance(api_version='v1')
    messages = privateca_base.GetMessagesModule(api_version='v1')
    test_response = client.projects_locations_caPools.TestIamPermissions(messages.PrivatecaProjectsLocationsCaPoolsTestIamPermissionsRequest(resource=issuing_ca_pool_ref.RelativeName(), testIamPermissionsRequest=messages.TestIamPermissionsRequest(permissions=_CERTIFICATE_CREATE_PERMISSIONS_ON_CA_POOL)))
    _CheckAllPermissions(test_response.permissions, _CERTIFICATE_CREATE_PERMISSIONS_ON_CA_POOL, 'issuing CA')