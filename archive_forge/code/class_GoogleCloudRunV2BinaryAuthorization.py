from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2BinaryAuthorization(_messages.Message):
    """Settings for Binary Authorization feature.

  Fields:
    breakglassJustification: Optional. If present, indicates to use Breakglass
      using this justification. If use_default is False, then it must be
      empty. For more information on breakglass, see
      https://cloud.google.com/binary-authorization/docs/using-breakglass
    policy: Optional. The path to a binary authorization policy. Format:
      projects/{project}/platforms/cloudRun/{policy-name}
    useDefault: Optional. If True, indicates to use the default project's
      binary authorization policy. If False, binary authorization will be
      disabled.
  """
    breakglassJustification = _messages.StringField(1)
    policy = _messages.StringField(2)
    useDefault = _messages.BooleanField(3)