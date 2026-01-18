from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def AddIncludeDenyFlag(parser):
    base.Argument('--include-deny', help='Include deny policies on the project and its ancestors in the result', action='store_true', default=False).AddToParser(parser)