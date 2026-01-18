from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Pubsub(base.Group):
    """Manage Cloud Pub/Sub topics, subscriptions, and snapshots."""
    category = base.DATA_ANALYTICS_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.DisableUserProjectQuota()