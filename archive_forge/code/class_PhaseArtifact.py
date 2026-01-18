from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PhaseArtifact(_messages.Message):
    """Contains the paths to the artifacts, relative to the URI, for a phase.

  Fields:
    jobManifestsPath: Output only. File path of the directory of rendered job
      manifests relative to the URI. This is only set if it is applicable.
    manifestPath: Output only. File path of the rendered manifest relative to
      the URI.
    skaffoldConfigPath: Output only. File path of the resolved Skaffold
      configuration relative to the URI.
  """
    jobManifestsPath = _messages.StringField(1)
    manifestPath = _messages.StringField(2)
    skaffoldConfigPath = _messages.StringField(3)