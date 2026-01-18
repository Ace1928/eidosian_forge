from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.api_lib.pubsub import topics
from googlecloudsdk.api_lib.util import exceptions as exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import times
import six
def ParsePushConfig(args, client=None):
    """Parses configs of push subscription from args."""
    push_endpoint = args.push_endpoint
    service_account_email = getattr(args, 'SERVICE_ACCOUNT_EMAIL', None)
    audience = getattr(args, 'OPTIONAL_AUDIENCE_OVERRIDE', None)
    if audience is not None and (push_endpoint is None or service_account_email is None):
        log.warning(PUSH_AUTH_TOKEN_AUDIENCE_MISSING_REQUIRED_FLAGS_WARNING.format(PUSH_ENDPOINT=push_endpoint or 'PUSH_ENDPOINT', SERVICE_ACCOUNT_EMAIL=service_account_email or 'SERVICE_ACCOUNT_EMAIL', OPTIONAL_AUDIENCE_OVERRIDE=audience))
    elif service_account_email is not None and push_endpoint is None:
        log.warning(PUSH_AUTH_SERVICE_ACCOUNT_MISSING_ENDPOINT_WARNING.format(SERVICE_ACCOUNT_EMAIL=service_account_email))
    if push_endpoint is None:
        if HasNoWrapper(args):
            raise InvalidArgumentError('argument --push-no-wrapper: --push-endpoint must be specified.')
        return None
    client = client or subscriptions.SubscriptionsClient()
    oidc_token = None
    if service_account_email is not None:
        oidc_token = client.messages.OidcToken(serviceAccountEmail=service_account_email, audience=audience)
    no_wrapper = None
    if HasNoWrapper(args):
        write_metadata = getattr(args, 'push_no_wrapper_write_metadata', False)
        no_wrapper = client.messages.NoWrapper(writeMetadata=write_metadata)
    return client.messages.PushConfig(pushEndpoint=push_endpoint, oidcToken=oidc_token, noWrapper=no_wrapper)