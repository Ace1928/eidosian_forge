import abc
import operator
import textwrap
import six
from apitools.base.protorpclite import descriptor as protorpc_descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import extra_types
class JsonEnumMapping(messages.Message):
    """Mapping from a python name to the wire name for an enum."""
    python_name = messages.StringField(1)
    json_name = messages.StringField(2)