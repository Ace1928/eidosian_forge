from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.alloydb import api_util
from googlecloudsdk.api_lib.alloydb import cluster_operations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.alloydb import cluster_helper
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ConstructCreateRequestFromArgs(self, alloydb_messages, location_ref, args):
    return cluster_helper.ConstructCreateRequestFromArgsAlpha(alloydb_messages, location_ref, args)