from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
class InvalidCustomConfigFileError(Error):
    """Error if a custom config file is improperly formatted."""