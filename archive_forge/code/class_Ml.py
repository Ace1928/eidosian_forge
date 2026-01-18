from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Ml(base.Group):
    """Use Google Cloud machine learning capabilities."""
    category = base.AI_AND_MACHINE_LEARNING_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args