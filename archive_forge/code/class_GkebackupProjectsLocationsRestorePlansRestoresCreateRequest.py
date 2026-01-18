from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsRestorePlansRestoresCreateRequest(_messages.Message):
    """A GkebackupProjectsLocationsRestorePlansRestoresCreateRequest object.

  Fields:
    parent: Required. The RestorePlan within which to create the Restore.
      Format: `projects/*/locations/*/restorePlans/*`
    restore: A Restore resource to be passed as the request body.
    restoreId: Required. The client-provided short name for the Restore
      resource. This name must: - be between 1 and 63 characters long
      (inclusive) - consist of only lower-case ASCII letters, numbers, and
      dashes - start with a lower-case letter - end with a lower-case letter
      or number - be unique within the set of Restores in this RestorePlan.
  """
    parent = _messages.StringField(1, required=True)
    restore = _messages.MessageField('Restore', 2)
    restoreId = _messages.StringField(3)