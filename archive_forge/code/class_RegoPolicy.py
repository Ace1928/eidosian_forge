from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegoPolicy(_messages.Message):
    """The rego policy defining this constraint template.

  Fields:
    libs: spec.targets.libs.
    policy: spec.targets.rego.
  """
    libs = _messages.StringField(1, repeated=True)
    policy = _messages.StringField(2)