from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeTemplateNodeTypeFlexibility(_messages.Message):
    """A NodeTemplateNodeTypeFlexibility object.

  Fields:
    cpus: A string attribute.
    localSsd: A string attribute.
    memory: A string attribute.
  """
    cpus = _messages.StringField(1)
    localSsd = _messages.StringField(2)
    memory = _messages.StringField(3)