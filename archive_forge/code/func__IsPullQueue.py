from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import parsers
def _IsPullQueue(r):
    return 'pullTarget' in r or ('type' in r and r['type'] == 'PULL')