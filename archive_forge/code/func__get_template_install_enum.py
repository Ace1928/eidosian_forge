from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.container.fleet.policycontroller import flags
def _get_template_install_enum(self, state: str) -> messages.Message:
    enums = self.messages.PolicyControllerTemplateLibraryConfig.InstallationValueValuesEnum
    enum = getattr(enums, state, None)
    if enum is None:
        raise exceptions.SafetyError('Invalid template install state: {}'.format(state))
    return enum