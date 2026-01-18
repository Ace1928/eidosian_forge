from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReimageInstanceRequest(_messages.Message):
    """Message requesting to perform reimage operation on a server.

  Fields:
    kmsKeyVersion: Optional. Name of the KMS crypto key version used to
      encrypt the initial passwords. The key has to have ASYMMETRIC_DECRYPT
      purpose. Format is `projects/{project}/locations/{location}/keyRings/{ke
      yring}/cryptoKeys/{key}/cryptoKeyVersions/{version}`.
    osImage: Required. The OS image code of the image which will be used in
      the reimage operation.
    sshKeys: Optional. List of SSH Keys used during reimaging an instance.
  """
    kmsKeyVersion = _messages.StringField(1)
    osImage = _messages.StringField(2)
    sshKeys = _messages.StringField(3, repeated=True)