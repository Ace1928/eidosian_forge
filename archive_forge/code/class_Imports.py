from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.Deprecate(is_removed=False, warning='This command has been deprecated. Please use `gcloud metastore services import` command group instead.')
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Imports(base.Group):
    """Manage metadata imports under Dataproc Metastore services."""