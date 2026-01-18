from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.beyondcorp.app import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.beyondcorp.app import util as command_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def ValidateAndParseAppEndpoint(unused_ref, args, request):
    """Validates app endpoint format and sets endpoint host and port after parsing.

  Args:
    unused_ref: The unused request URL.
    args: arguments set by user.
    request: create connection request raised by framework.

  Returns:
    request with modified application endpoint host and port argument.

  Raises:
    ApplicationEndpointParseError:
  """
    if args.IsSpecified('application_endpoint'):
        endpoint_array = args.application_endpoint.split(':')
        if len(endpoint_array) == 2 and endpoint_array[1].isdigit():
            messages = api_util.GetMessagesModule(args.calliope_command.ReleaseTrack())
            app_connection = GetVersionedConnectionReq(args, request)
            if app_connection is None:
                app_connection = GetVersionedConnectionMsg(args, messages)()
            if app_connection.applicationEndpoint is None:
                app_connection.applicationEndpoint = GetVersionedEndpointMsg(args, messages)()
            app_connection.applicationEndpoint.host = endpoint_array[0]
            app_connection.applicationEndpoint.port = int(endpoint_array[1])
            if args.calliope_command.ReleaseTrack() == base.ReleaseTrack.ALPHA:
                request.googleCloudBeyondcorpAppconnectionsV1alphaAppConnection = app_connection
            else:
                request.googleCloudBeyondcorpAppconnectionsV1AppConnection = app_connection
        else:
            raise ApplicationEndpointParseError(APP_ENDPOINT_PARSE_ERROR.format(args.application_endpoint))
    return request