from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StepTemplate(_messages.Message):
    """StepTemplate can be used as the basis for all step containers within the
  Task, so that the steps inherit settings on the base container.

  Fields:
    env: Optional. List of environment variables to set in the Step. Cannot be
      updated.
    volumeMounts: Optional. Pod volumes to mount into the container's
      filesystem.
  """
    env = _messages.MessageField('EnvVar', 1, repeated=True)
    volumeMounts = _messages.MessageField('VolumeMount', 2, repeated=True)