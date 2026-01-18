from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3FormParameter(_messages.Message):
    """Represents a form parameter.

  Fields:
    advancedSettings: Hierarchical advanced settings for this parameter. The
      settings exposed at the lower level overrides the settings exposed at
      the higher level.
    defaultValue: The default value of an optional parameter. If the parameter
      is required, the default value will be ignored.
    displayName: Required. The human-readable name of the parameter, unique
      within the form.
    entityType: Required. The entity type of the parameter. Format:
      `projects/-/locations/-/agents/-/entityTypes/` for system entity types
      (for example, `projects/-/locations/-/agents/-/entityTypes/sys.date`),
      or `projects//locations//agents//entityTypes/` for developer entity
      types.
    fillBehavior: Required. Defines fill behavior for the parameter.
    isList: Indicates whether the parameter represents a list of values.
    redact: Indicates whether the parameter content should be redacted in log.
      If redaction is enabled, the parameter content will be replaced by
      parameter name during logging. Note: the parameter content is subject to
      redaction if either parameter level redaction or entity type level
      redaction is enabled.
    required: Indicates whether the parameter is required. Optional parameters
      will not trigger prompts; however, they are filled if the user specifies
      them. Required parameters must be filled before form filling concludes.
  """
    advancedSettings = _messages.MessageField('GoogleCloudDialogflowCxV3AdvancedSettings', 1)
    defaultValue = _messages.MessageField('extra_types.JsonValue', 2)
    displayName = _messages.StringField(3)
    entityType = _messages.StringField(4)
    fillBehavior = _messages.MessageField('GoogleCloudDialogflowCxV3FormParameterFillBehavior', 5)
    isList = _messages.BooleanField(6)
    redact = _messages.BooleanField(7)
    required = _messages.BooleanField(8)