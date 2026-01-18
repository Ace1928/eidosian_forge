from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1Volume(_messages.Message):
    """Volume describes a Docker container volume which is mounted into build
  steps in order to persist files across build step execution.

  Fields:
    name: Name of the volume to mount. Volume names must be unique per build
      step and must be valid names for Docker volumes. Each named volume must
      be used by at least two build steps.
    path: Path at which to mount the volume. Paths must be absolute and cannot
      conflict with other volume paths on the same build step or with certain
      reserved volume paths.
  """
    name = _messages.StringField(1)
    path = _messages.StringField(2)