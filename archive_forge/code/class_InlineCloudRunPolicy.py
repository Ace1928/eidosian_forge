from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InlineCloudRunPolicy(_messages.Message):
    """A binary authorization policy for Cloud Run deployments.

  Fields:
    allowlistPatterns: Optional. List of images that will be allowed
      regardless of the platform-based policies. Allowlists are always
      evaluated prior to evaluating any platform-based policies. An image name
      pattern to allowlist is in the form `registry/path/to/image`. A trailing
      `*` is supported as a wildcard, but this is allowed only in text after
      the `registry/` part.
    rule: Required. The evaluation rule used for evaluating a Cloud Run
      deployment.
  """
    allowlistPatterns = _messages.MessageField('AllowlistPattern', 1, repeated=True)
    rule = _messages.MessageField('EvaluationRule', 2)