from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import clusters
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import log
def _Cluster(self, args):
    msgs = util.GetAdminMessages()
    storage_type = msgs.Cluster.DefaultStorageTypeValueValuesEnum.STORAGE_TYPE_UNSPECIFIED
    cluster = msgs.Cluster(serveNodes=args.num_nodes, location=util.LocationUrl(args.zone), defaultStorageType=storage_type)
    kms_key = arguments.GetAndValidateKmsKeyName(args)
    if kms_key:
        cluster.encryptionConfig = msgs.EncryptionConfig(kmsKeyName=kms_key)
    if args.autoscaling_min_nodes is not None or args.autoscaling_max_nodes is not None or args.autoscaling_cpu_target is not None or (args.autoscaling_storage_target is not None):
        cluster.clusterConfig = clusters.BuildClusterConfig(autoscaling_min=args.autoscaling_min_nodes, autoscaling_max=args.autoscaling_max_nodes, autoscaling_cpu_target=args.autoscaling_cpu_target, autoscaling_storage_target=args.autoscaling_storage_target)
        cluster.serveNodes = None
    return cluster