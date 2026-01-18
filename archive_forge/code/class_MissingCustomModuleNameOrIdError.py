from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
class MissingCustomModuleNameOrIdError(Error):
    """An error representing a missing custom module name or id."""

    def __init__(self):
        super(Error, self).__init__('Missing custom module name or ID.')