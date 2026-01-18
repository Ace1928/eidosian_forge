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
def _register_legacy_retries(self, client):
    endpoint_prefix = client.meta.service_model.endpoint_prefix
    service_id = client.meta.service_model.service_id
    service_event_name = service_id.hyphenize()
    original_config = self._loader.load_data('_retry')
    if not original_config:
        return
    retries = self._transform_legacy_retries(client.meta.config.retries)
    retry_config = self._retry_config_translator.build_retry_config(endpoint_prefix, original_config.get('retry', {}), original_config.get('definitions', {}), retries)
    logger.debug('Registering retry handlers for service: %s', client.meta.service_model.service_name)
    handler = self._retry_handler_factory.create_retry_handler(retry_config, endpoint_prefix)
    unique_id = 'retry-config-%s' % service_event_name
    client.meta.events.register(f'needs-retry.{service_event_name}', handler, unique_id=unique_id)