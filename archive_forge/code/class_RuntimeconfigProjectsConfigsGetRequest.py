from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeconfigProjectsConfigsGetRequest(_messages.Message):
    """A RuntimeconfigProjectsConfigsGetRequest object.

  Fields:
    name: The name of the RuntimeConfig resource to retrieve, in the format:
      `projects/[PROJECT_ID]/configs/[CONFIG_NAME]`
  """
    name = _messages.StringField(1, required=True)