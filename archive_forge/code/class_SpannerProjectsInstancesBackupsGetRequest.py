from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesBackupsGetRequest(_messages.Message):
    """A SpannerProjectsInstancesBackupsGetRequest object.

  Fields:
    name: Required. Name of the backup. Values are of the form
      `projects//instances//backups/`.
  """
    name = _messages.StringField(1, required=True)