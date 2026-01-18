from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class Datafusion(base.Group):
    """Create and manage Cloud Data Fusion Instances.


    Cloud Data Fusion is a fully managed, cloud-native data integration service
    that helps users efficiently build and manage ETL/ELT data pipelines. With
    a graphical interface and a broad open-source library of preconfigured
    connectors and transformations, Data Fusion shifts an
    organization's focus away from code and integration to insights and action.

    ## EXAMPLES

    To see how to create and manage instances, run:

        $ {command} instances --help

    To see how to manage long-running operations, run:

        $ {command} operations --help
  """
    category = base.BIG_DATA_CATEGORY

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args
        base.DisableUserProjectQuota()