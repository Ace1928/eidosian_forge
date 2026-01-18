from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.binauthz import apis
from googlecloudsdk.api_lib.container.binauthz import util
from googlecloudsdk.command_lib.container.binauthz import exceptions
from googlecloudsdk.command_lib.kms import maps as kms_maps
def AddPkixKey(self, attestor_ref, pkix_pubkey_content, pkix_sig_algorithm, id_override=None, comment=None):
    """Add a key to an attestor.

    Args:
      attestor_ref: ResourceSpec, The attestor to be updated.
      pkix_pubkey_content: The PEM-encoded PKIX public key.
      pkix_sig_algorithm: The PKIX public key signature algorithm.
      id_override: If provided, the key ID to use instead of the API-generated
          one.
      comment: The comment on the public key.

    Returns:
      The added public key.

    Raises:
      AlreadyExistsError: If a public key with the same key content was found on
          the attestor.
    """
    attestor = self.Get(attestor_ref)
    existing_ids = set((public_key.id for public_key in self.GetNoteAttr(attestor).publicKeys))
    if id_override is not None and id_override in existing_ids:
        raise exceptions.AlreadyExistsError('Public key with ID [{}] already present on attestor [{}]'.format(id_override, attestor.name))
    self.GetNoteAttr(attestor).publicKeys.append(self.messages.AttestorPublicKey(id=id_override, pkixPublicKey=self.messages.PkixPublicKey(publicKeyPem=pkix_pubkey_content, signatureAlgorithm=pkix_sig_algorithm), comment=comment))
    updated_attestor = self.client.projects_attestors.Update(attestor)
    return next((public_key for public_key in self.GetNoteAttr(updated_attestor).publicKeys if public_key.id not in existing_ids))