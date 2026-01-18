from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidateExclusivityResponse(_messages.Message):
    """The response of exclusivity artifacts validation result status.

  Fields:
    status: The validation result. * `OK` means that exclusivity is validated,
      assuming the manifest produced by GenerateExclusivityManifest is
      successfully applied. * `ALREADY_EXISTS` means that the Membership CRD
      is already owned by another Hub. See `status.message` for more
      information.
  """
    status = _messages.MessageField('GoogleRpcStatus', 1)