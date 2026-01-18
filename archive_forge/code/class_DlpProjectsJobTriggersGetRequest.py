from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsJobTriggersGetRequest(_messages.Message):
    """A DlpProjectsJobTriggersGetRequest object.

  Fields:
    name: Required. Resource name of the project and the triggeredJob, for
      example `projects/dlp-test-project/jobTriggers/53234423`.
  """
    name = _messages.StringField(1, required=True)