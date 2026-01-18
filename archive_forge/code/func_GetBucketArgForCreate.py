from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def GetBucketArgForCreate():
    return base.Argument('--buckets', type=arg_parsers.ArgList(), metavar='BUCKET', help='A list of buckets to crawl. This argument should be provided if and only if `--crawl-scope=BUCKET` was specified.')