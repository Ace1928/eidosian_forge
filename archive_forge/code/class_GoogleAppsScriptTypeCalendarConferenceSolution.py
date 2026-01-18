from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsScriptTypeCalendarConferenceSolution(_messages.Message):
    """Defines conference related values.

  Fields:
    id: Required. IDs should be uniquely assigned across conference solutions
      within one add-on, otherwise the wrong conference solution might be used
      when the add-on is triggered. While you can change the display name of
      an add-on, the ID shouldn't be changed.
    logoUrl: Required. The URL for the logo image of the conference solution.
    name: Required. The display name of the conference solution.
    onCreateFunction: Required. The endpoint to call when conference data
      should be created.
  """
    id = _messages.StringField(1)
    logoUrl = _messages.StringField(2)
    name = _messages.StringField(3)
    onCreateFunction = _messages.StringField(4)