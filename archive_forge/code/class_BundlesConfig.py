from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BundlesConfig(_messages.Message):
    """Configuration for the bundles that can be enabled on the KrmApiHost.
  Bundles not ready for public consumption must have a visibility label. e.g.:
  YakimaConfig yakima_config = 2 [(google.api.field_visibility).restriction =
  "GOOGLE_INTERNAL, YAKIMA_TRUSTED_TESTER, GCLOUD_TESTER"];

  Fields:
    configControllerConfig: Configuration for the Config Controller bundle.
  """
    configControllerConfig = _messages.MessageField('ConfigControllerConfig', 1)