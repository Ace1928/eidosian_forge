from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1SecretManagerSecret(_messages.Message):
    """Pairs a secret environment variable with a SecretVersion in Secret
  Manager.

  Fields:
    env: Environment variable name to associate with the secret. Secret
      environment variables must be unique across all of a build's secrets,
      and must be used by at least one build step.
    versionName: Resource name of the SecretVersion. In format:
      projects/*/secrets/*/versions/*
  """
    env = _messages.StringField(1)
    versionName = _messages.StringField(2)