from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.binauthz import apis
from googlecloudsdk.api_lib.container.binauthz import util
from googlecloudsdk.command_lib.container.binauthz import exceptions
from googlecloudsdk.command_lib.kms import maps as kms_maps
def AddPgpKey(self, attestor_ref, pgp_pubkey_content, comment=None):
    """Add a PGP key to an attestor.

    Args:
      attestor_ref: ResourceSpec, The attestor to be updated.
      pgp_pubkey_content: The contents of the PGP public key file.
      comment: The comment on the public key.

    Returns:
      The added public key.

    Raises:
      AlreadyExistsError: If a public key with the same key content was found on
          the attestor.
    """
    attestor = self.Get(attestor_ref)
    existing_pub_keys = set((public_key.asciiArmoredPgpPublicKey for public_key in self.GetNoteAttr(attestor).publicKeys))
    if pgp_pubkey_content in existing_pub_keys:
        raise exceptions.AlreadyExistsError('Provided public key already present on attestor [{}]'.format(attestor.name))
    existing_ids = set((public_key.id for public_key in self.GetNoteAttr(attestor).publicKeys))
    self.GetNoteAttr(attestor).publicKeys.append(self.messages.AttestorPublicKey(asciiArmoredPgpPublicKey=pgp_pubkey_content, comment=comment))
    updated_attestor = self.client.projects_attestors.Update(attestor)
    return next((public_key for public_key in self.GetNoteAttr(updated_attestor).publicKeys if public_key.id not in existing_ids))