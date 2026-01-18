from googlecloudsdk.command_lib.scc.manage import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
class InvalidResourceFileError(Error):
    """Error if a test data file is improperly formatted."""