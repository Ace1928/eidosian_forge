from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestEnabledRequest(_messages.Message):
    """The request to test a value against the result of merging consumer
  policies in the resource hierarchy.

  Fields:
    serviceName: The name of a service to test for enablement.
  """
    serviceName = _messages.StringField(1)