from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.endpoints import services_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
class ProducerServiceCompleter(completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(ProducerServiceCompleter, self).__init__(collection=services_util.SERVICES_COLLECTION, list_command=_SERVICES_LIST_COMMAND, flags=['produced'], **kwargs)