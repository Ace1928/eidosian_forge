from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1IntentParameter(_messages.Message):
    """Represents an intent parameter.

  Fields:
    entityType: Required. The entity type of the parameter. Format:
      `projects/-/locations/-/agents/-/entityTypes/` for system entity types
      (for example, `projects/-/locations/-/agents/-/entityTypes/sys.date`),
      or `projects//locations//agents//entityTypes/` for developer entity
      types.
    id: Required. The unique identifier of the parameter. This field is used
      by training phrases to annotate their parts.
    isList: Indicates whether the parameter represents a list of values.
    redact: Indicates whether the parameter content should be redacted in log.
      If redaction is enabled, the parameter content will be replaced by
      parameter name during logging. Note: the parameter content is subject to
      redaction if either parameter level redaction or entity type level
      redaction is enabled.
  """
    entityType = _messages.StringField(1)
    id = _messages.StringField(2)
    isList = _messages.BooleanField(3)
    redact = _messages.BooleanField(4)