from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CsmSettings(_messages.Message):
    """Configuration for RCToken generated for service mesh workloads protected
  by IAP. RCToken are IAP generated JWTs that can be verified at the
  application. The RCToken is primarily used for service mesh deployments, and
  can be scoped to a single mesh by configuring the audience field
  accordingly.

  Fields:
    rctokenAud: Audience claim set in the generated RCToken. This value is not
      validated by IAP.
  """
    rctokenAud = _messages.StringField(1)