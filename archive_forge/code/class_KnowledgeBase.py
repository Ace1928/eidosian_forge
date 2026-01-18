from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KnowledgeBase(_messages.Message):
    """A KnowledgeBase object.

  Fields:
    name: The KB name (generally of the form KB[0-9]+ (e.g., KB123456)).
    url: A link to the KB in the [Windows update catalog]
      (https://www.catalog.update.microsoft.com/).
  """
    name = _messages.StringField(1)
    url = _messages.StringField(2)