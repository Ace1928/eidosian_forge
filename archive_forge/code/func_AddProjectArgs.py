from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
def AddProjectArgs(parser):
    parser.add_argument('--project', metavar='PROJECT_ID', required=True, help='The project ID or number to perform the analysis.')