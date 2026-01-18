from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
class InvalidEnablementStateError(Error):
    """Error if an enablement state is anything but ENABLED or DISABLED."""