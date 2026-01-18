from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def GetBucketArgsForUpdate():
    """Returns bucket-related args for crawler update."""
    bucket_group = base.ArgumentGroup(help='Update buckets to crawl. These arguments can be provided only if the crawler will be bucket-scoped after updating.')
    bucket_group.AddArgument(base.Argument('--add-buckets', type=arg_parsers.ArgList(), metavar='BUCKET', help='List of buckets to add to the crawl scope.'))
    remove_bucket_group = base.ArgumentGroup(mutex=True)
    remove_bucket_group.AddArgument(base.Argument('--remove-buckets', type=arg_parsers.ArgList(), metavar='BUCKET', help='List of buckets to remove from the crawl scope.'))
    remove_bucket_group.AddArgument(base.Argument('--clear-buckets', action='store_true', help='If specified, clear the existing list of buckets in the crawl scope.'))
    bucket_group.AddArgument(remove_bucket_group)
    return bucket_group