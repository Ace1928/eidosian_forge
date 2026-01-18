from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class BigtableV2(base.Group):
    """Manage your Cloud Bigtable storage."""
    category = base.DATABASES_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.DisableUserProjectQuota()