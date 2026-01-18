from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import parsers
def _IsPushQueue(r):
    return 'appEngineHttpTarget' in r or 'appEngineHttpQueue' in r or 'appEngineRoutingOverride' in r or ('type' in r and r['type'] == 'PUSH')