from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core import exceptions
class ServiceBindingsCompleter(completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(ServiceBindingsCompleter, self).__init__(collection='networkservices.projects.locations.serviceBindings', list_command='network-services service-bindings list --location=global --uri', **kwargs)