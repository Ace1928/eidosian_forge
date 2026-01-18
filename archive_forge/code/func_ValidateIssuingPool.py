from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
def ValidateIssuingPool(ca_pool_name, issuing_ca_id):
    """Checks that a CA Pool is valid to be issuing Pool for a subordinate.

  Args:
    ca_pool_name: The resource name of the issuing CA Pool.
    issuing_ca_id: The optional CA ID in the CA Pool to validate.

  Raises:
    InvalidArgumentException if the CA Pool does not exist or is not enabled.
  """
    try:
        client = privateca_base.GetClientInstance(api_version='v1')
        messages = privateca_base.GetMessagesModule(api_version='v1')
        enabled_state = messages.CertificateAuthority.StateValueValuesEnum.ENABLED
        ca_list_response = client.projects_locations_caPools_certificateAuthorities.List(messages.PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesListRequest(parent=ca_pool_name))
        ca_list = ca_list_response.certificateAuthorities
        if issuing_ca_id:
            _ValidateIssuingCa(ca_pool_name, issuing_ca_id, ca_list)
            return
        ca_states = [ca.state for ca in ca_list]
        if enabled_state not in ca_states:
            raise exceptions.InvalidArgumentException('--issuer-pool', 'The issuing CA Pool [{}] did not have any CAs in ENABLED state of the {} CAs found. Please create or enable a CA and try again.'.format(ca_pool_name, len(ca_list)))
    except apitools_exceptions.HttpNotFoundError:
        raise exceptions.InvalidArgumentException('--issuer-pool', 'The issuing CA Pool [{}] was not found. Please verify this information is correct and try again.'.format(ca_pool_name))