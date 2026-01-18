from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
class KeyRingCompleter(ListCommandCompleter):

    def __init__(self, **kwargs):
        super(KeyRingCompleter, self).__init__(collection=KEY_RING_COLLECTION, list_command='kms keyrings list --uri', flags=['location'], **kwargs)