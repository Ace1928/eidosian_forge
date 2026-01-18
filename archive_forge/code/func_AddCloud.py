from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
def AddCloud(self):
    self._AddFlag('--cloud', default=False, action='store_true', hidden=True, help='deploy code to Cloud Run')
    self._AddFlag('--region', help='region to deploy the dev service', hidden=True)