from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
class InvalidUpdateMaskInputError(Error):
    """Error if neither a custom configuration or an enablement state were given to update."""