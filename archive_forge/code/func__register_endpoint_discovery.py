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
def _register_endpoint_discovery(self, client, endpoint_url, config):
    if endpoint_url is not None:
        return
    if client.meta.service_model.endpoint_discovery_operation is None:
        return
    events = client.meta.events
    service_id = client.meta.service_model.service_id.hyphenize()
    enabled = False
    if config and config.endpoint_discovery_enabled is not None:
        enabled = config.endpoint_discovery_enabled
    elif self._config_store:
        enabled = self._config_store.get_config_variable('endpoint_discovery_enabled')
    enabled = self._normalize_endpoint_discovery_config(enabled)
    if enabled and self._requires_endpoint_discovery(client, enabled):
        discover = enabled is True
        manager = EndpointDiscoveryManager(client, always_discover=discover)
        handler = EndpointDiscoveryHandler(manager)
        handler.register(events, service_id)
    else:
        events.register('before-parameter-build', block_endpoint_discovery_required_operations)