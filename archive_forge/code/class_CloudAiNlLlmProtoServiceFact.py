from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServiceFact(_messages.Message):
    """A condense version of WorldFact
  (assistant/boq/lamda/factuality/proto/factuality.proto) to propagate the
  essential information about the fact used in factuality to the upstream
  caller.

  Fields:
    query: Query that is used to retrieve this fact.
    summary: If present, the summary/snippet of the fact.
    title: If present, it refers to the title of this fact.
    url: If present, this URL links to the webpage of the fact.
  """
    query = _messages.StringField(1)
    summary = _messages.StringField(2)
    title = _messages.StringField(3)
    url = _messages.StringField(4)