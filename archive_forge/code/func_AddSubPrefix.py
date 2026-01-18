from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core.exceptions import Error
def AddSubPrefix(self, pdp_ref, name, ip_cidr_range, description, delegatee_project, is_addresses):
    """Adds a delegated sub prefix to public delegated prefix using PATCH.

    Args:
      pdp_ref: resource reference.
      name: sub prefix name.
      ip_cidr_range: sub prefix IP address range.
      description: sub prefix description.
      delegatee_project: sub prefix target project.
      is_addresses: sub prefix isAddress parameter.

    Returns:
      Operation result from the poller.

    Raises:
      PublicDelegatedPrefixPatchError:
        when delegated prefix already has a sub prefix with specified name.
    """
    resource = self.Get(pdp_ref)
    for sub_prefix in resource.publicDelegatedSubPrefixs:
        if sub_prefix.name == name:
            raise PublicDelegatedPrefixPatchError('Delegated sub prefix [{}] already exists in public delegated prefix [{}]'.format(name, pdp_ref.Name()))
    resource.publicDelegatedSubPrefixs.append(self.messages.PublicDelegatedPrefixPublicDelegatedSubPrefix(name=name, description=description, ipCidrRange=ip_cidr_range, delegateeProject=delegatee_project, isAddress=is_addresses))
    return self._Patch(pdp_ref, resource)