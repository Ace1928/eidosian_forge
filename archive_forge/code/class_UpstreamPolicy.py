from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpstreamPolicy(_messages.Message):
    """Artifact policy configuration for the repository contents.

  Fields:
    id: The user-provided ID of the upstream policy.
    priority: Entries with a greater priority value take precedence in the
      pull order.
    repository: A reference to the repository resource, for example:
      `projects/p1/locations/us-central1/repositories/repo1`.
  """
    id = _messages.StringField(1)
    priority = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    repository = _messages.StringField(3)