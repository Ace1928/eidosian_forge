from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
import six
def AddNoCacheFlag(parser, hidden=False):
    """Add a flag to disable layer caching."""
    parser.add_argument('--no-cache', hidden=hidden, action='store_true', help='If set, disable layer caching when building with Kaniko.\n\nThis has the same effect as setting the builds/kaniko_cache_ttl property to 0 for this build.  This can be useful in cases where Dockerfile builds are non-deterministic and a non-deterministic result should not be cached.')