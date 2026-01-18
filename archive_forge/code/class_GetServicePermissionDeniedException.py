from googlecloudsdk.api_lib.util import exceptions as api_lib_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
class GetServicePermissionDeniedException(Error):
    """Permission denied exception for get service command."""