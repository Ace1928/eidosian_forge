from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2InputConfig(_messages.Message):
    """Represents the configuration of importing a set of conversation files in
  Google Cloud Storage.

  Fields:
    gcsSource: The Cloud Storage URI has the form gs:////agent*.json.
      Wildcards are allowed and will be expanded into all matched JSON files,
      which will be read as one conversation per file.
  """
    gcsSource = _messages.MessageField('GoogleCloudDialogflowV2GcsSources', 1)