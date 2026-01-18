from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListModulesResponse(_messages.Message):
    """Response message for ModulesService.ListModules.

  Fields:
    modules: The list of modules.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    unreachable: Locations that could not be reached.
  """
    modules = _messages.MessageField('Module', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)