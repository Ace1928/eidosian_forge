from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.container.gkemulticloud import operations as op_api_util
from googlecloudsdk.api_lib.container.gkemulticloud import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def ClientMessage(name, action=None, region=None):
    msg = 'client [{name}]'.format(name=name)
    if action:
        msg = '{action} '.format(action=action) + msg
    if region:
        msg += ' in [{region}]'.format(region=region)
    return msg