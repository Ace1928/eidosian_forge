from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.binauthz import apis
from googlecloudsdk.api_lib.container.binauthz import util
from googlecloudsdk.command_lib.container.binauthz import exceptions
from googlecloudsdk.command_lib.kms import maps as kms_maps
def RemoveKey(self, attestor_ref, pubkey_id):
    """Remove a key on an attestor.

    Args:
      attestor_ref: ResourceSpec, The attestor to be updated.
      pubkey_id: The ID of the key to remove.

    Raises:
      NotFoundError: If an expected public key could not be located by ID.
    """
    attestor = self.Get(attestor_ref)
    existing_ids = set((public_key.id for public_key in self.GetNoteAttr(attestor).publicKeys))
    if pubkey_id not in existing_ids:
        raise exceptions.NotFoundError('No matching public key found on attestor [{}]'.format(attestor.name))
    self.GetNoteAttr(attestor).publicKeys = [public_key for public_key in self.GetNoteAttr(attestor).publicKeys if public_key.id != pubkey_id]
    self.client.projects_attestors.Update(attestor)