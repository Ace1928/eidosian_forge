from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.cloudbuild import build
@classmethod
def MakeBuildIdArtifact(cls, build_id):
    return cls(cls.BuildType.BUILD_ID, build_id)