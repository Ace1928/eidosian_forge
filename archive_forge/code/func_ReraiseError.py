from googlecloudsdk.api_lib.util import exceptions as api_lib_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
def ReraiseError(err, klass):
    """Transform and re-raise error helper."""
    core_exceptions.reraise(klass(api_lib_exceptions.HttpException(err)))