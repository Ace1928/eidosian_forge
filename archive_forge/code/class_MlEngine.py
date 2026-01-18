from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA, base.ReleaseTrack.BETA)
class MlEngine(base.Group):
    """Manage AI Platform jobs and models.

  The {command} command group lets you manage AI Platform jobs and
  training models.

  AI Platform is a managed service that enables you to easily build
  machine
  learning models, that work on any type of data, of any size. Create your model
  with the powerful TensorFlow framework that powers many Google products, from
  Google Photos to Google Cloud Speech.

  More information on AI Platform can be found here:
  https://cloud.google.com/ml
  and detailed documentation can be found here:
  https://cloud.google.com/ml/docs/
  """
    category = base.AI_AND_MACHINE_LEARNING_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.DisableUserProjectQuota()
        resources.REGISTRY.RegisterApiByName('ml', 'v1')