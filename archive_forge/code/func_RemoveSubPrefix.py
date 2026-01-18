from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core.exceptions import Error
def RemoveSubPrefix(self, pdp_ref, name):
    """Removes a delegated sub prefix from public delegated prefix using PATCH.

    Args:
      pdp_ref: resource reference.
      name: name of sub prefix to remove.

    Returns:
      Operation result from the poller.

    Raises:
      PublicDelegatedPrefixPatchError:
        when delegated prefix does not have a sub prefix with specified name.
    """
    resource = self.Get(pdp_ref)
    index_to_remove = None
    for i, sub_prefix in enumerate(resource.publicDelegatedSubPrefixs):
        if sub_prefix.name == name:
            index_to_remove = i
    if index_to_remove is None:
        raise PublicDelegatedPrefixPatchError('Delegated sub prefix [{}] does not exist in public delegated prefix [{}]'.format(name, pdp_ref.Name()))
    resource.publicDelegatedSubPrefixs.pop(index_to_remove)
    return self._Patch(pdp_ref, resource)