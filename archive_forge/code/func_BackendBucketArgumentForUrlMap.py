from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def BackendBucketArgumentForUrlMap(required=True):
    return compute_flags.ResourceArgument(resource_name='backend bucket', name='--default-backend-bucket', required=required, completer=BackendBucketsCompleter, global_collection='compute.backendBuckets')