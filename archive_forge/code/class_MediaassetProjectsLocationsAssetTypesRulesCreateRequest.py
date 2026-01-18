from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsAssetTypesRulesCreateRequest(_messages.Message):
    """A MediaassetProjectsLocationsAssetTypesRulesCreateRequest object.

  Fields:
    parent: Required. The parent resource where this Rule will be created.
      Format: `projects/{project}/locations/{location}/assetTypes/{type}`
    rule: A Rule resource to be passed as the request body.
    ruleId: Required. The ID to use for the Rule, which will become the final
      component of the rule's resource name. This value should be 4-63
      characters, and valid characters are /a-z-/.
  """
    parent = _messages.StringField(1, required=True)
    rule = _messages.MessageField('Rule', 2)
    ruleId = _messages.StringField(3)