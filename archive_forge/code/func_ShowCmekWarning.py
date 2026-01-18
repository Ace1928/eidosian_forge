from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import instance_prop_reducers as reducers
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def ShowCmekWarning(resource_type_label, instance_type_label=None):
    if instance_type_label is None:
        log.warning('Your {} will be encrypted with a customer-managed key. If anyone destroys this key, all data encrypted with it will be permanently lost.'.format(resource_type_label))
    else:
        log.warning("Your {} will be encrypted with {}'s customer-managed encryption key. If anyone destroys this key, all data encrypted with it will be permanently lost.".format(resource_type_label, instance_type_label))