import logging
from botocore import waiter, xform_name
from botocore.args import ClientArgsCreator
from botocore.auth import AUTH_TYPE_MAPS
from botocore.awsrequest import prepare_request_dict
from botocore.compress import maybe_compress_request
from botocore.config import Config
from botocore.credentials import RefreshableCredentials
from botocore.discovery import (
from botocore.docs.docstring import ClientMethodDocstring, PaginatorDocstring
from botocore.exceptions import (
from botocore.history import get_global_history_recorder
from botocore.hooks import first_non_none_response
from botocore.httpchecksum import (
from botocore.model import ServiceModel
from botocore.paginate import Paginator
from botocore.retries import adaptive, standard
from botocore.useragent import UserAgentString
from botocore.utils import (
from botocore.exceptions import ClientError  # noqa
from botocore.utils import S3ArnParamHandler  # noqa
from botocore.utils import S3ControlArnParamHandler  # noqa
from botocore.utils import S3ControlEndpointSetter  # noqa
from botocore.utils import S3EndpointSetter  # noqa
from botocore.utils import S3RegionRedirector  # noqa
from botocore import UNSIGNED  # noqa
def _resolve_endpoint_ruleset(self, operation_model, params, request_context, ignore_signing_region=False):
    """Returns endpoint URL and list of additional headers returned from
        EndpointRulesetResolver for the given operation and params. If the
        ruleset resolver is not available, for example because the service has
        no endpoints ruleset file, the legacy endpoint resolver's value is
        returned.

        Use ignore_signing_region for generating presigned URLs or any other
        situation where the signing region information from the ruleset
        resolver should be ignored.

        Returns tuple of URL and headers dictionary. Additionally, the
        request_context dict is modified in place with any signing information
        returned from the ruleset resolver.
        """
    if self._ruleset_resolver is None:
        endpoint_url = self.meta.endpoint_url
        additional_headers = {}
        endpoint_properties = {}
    else:
        endpoint_info = self._ruleset_resolver.construct_endpoint(operation_model=operation_model, call_args=params, request_context=request_context)
        endpoint_url = endpoint_info.url
        additional_headers = endpoint_info.headers
        endpoint_properties = endpoint_info.properties
        auth_schemes = endpoint_info.properties.get('authSchemes')
        if auth_schemes is not None:
            auth_info = self._ruleset_resolver.auth_schemes_to_signing_ctx(auth_schemes)
            auth_type, signing_context = auth_info
            request_context['auth_type'] = auth_type
            if 'region' in signing_context and ignore_signing_region:
                del signing_context['region']
            if 'signing' in request_context:
                request_context['signing'].update(signing_context)
            else:
                request_context['signing'] = signing_context
    return (endpoint_url, additional_headers, endpoint_properties)