from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class TensorboardExperiment(base.Group):
    """Manage Vertex AI Tensorboard experiments."""
    category = base.VERTEX_AI_CATEGORY