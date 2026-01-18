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
class ClientMeta:
    """Holds additional client methods.

    This class holds additional information for clients.  It exists for
    two reasons:

        * To give advanced functionality to clients
        * To namespace additional client attributes from the operation
          names which are mapped to methods at runtime.  This avoids
          ever running into collisions with operation names.

    """

    def __init__(self, events, client_config, endpoint_url, service_model, method_to_api_mapping, partition):
        self.events = events
        self._client_config = client_config
        self._endpoint_url = endpoint_url
        self._service_model = service_model
        self._method_to_api_mapping = method_to_api_mapping
        self._partition = partition

    @property
    def service_model(self):
        return self._service_model

    @property
    def region_name(self):
        return self._client_config.region_name

    @property
    def endpoint_url(self):
        return self._endpoint_url

    @property
    def config(self):
        return self._client_config

    @property
    def method_to_api_mapping(self):
        return self._method_to_api_mapping

    @property
    def partition(self):
        return self._partition