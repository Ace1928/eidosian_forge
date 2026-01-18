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
def _create_client_class(self, service_name, service_model):
    class_attributes = self._create_methods(service_model)
    py_name_to_operation_name = self._create_name_mapping(service_model)
    class_attributes['_PY_TO_OP_NAME'] = py_name_to_operation_name
    bases = [BaseClient]
    service_id = service_model.service_id.hyphenize()
    self._event_emitter.emit('creating-client-class.%s' % service_id, class_attributes=class_attributes, base_classes=bases)
    class_name = get_service_module_name(service_model)
    cls = type(str(class_name), tuple(bases), class_attributes)
    return cls