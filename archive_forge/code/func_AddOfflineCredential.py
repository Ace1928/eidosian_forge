from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddOfflineCredential(parser):
    parser.add_argument('--offline-credential', action='store_true', help='\n      Once specified, an offline credential will be generated for the cluster.\n      ')