from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ContainerOverride(_messages.Message):
    """Per-container override specification.

  Fields:
    args: Optional. Arguments to the entrypoint. Will replace existing args
      for override.
    clearArgs: Optional. True if the intention is to clear out existing args
      list.
    env: List of environment variables to set in the container. Will be merged
      with existing env for override.
    name: The name of the container specified as a DNS_LABEL.
  """
    args = _messages.StringField(1, repeated=True)
    clearArgs = _messages.BooleanField(2)
    env = _messages.MessageField('GoogleCloudRunV2EnvVar', 3, repeated=True)
    name = _messages.StringField(4)