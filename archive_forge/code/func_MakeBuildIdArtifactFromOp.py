from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.cloudbuild import build
@classmethod
def MakeBuildIdArtifactFromOp(cls, build_op):
    build_id = build.GetBuildProp(build_op, 'id', required=True)
    return cls(cls.BuildType.BUILD_ID, build_id, build_op)