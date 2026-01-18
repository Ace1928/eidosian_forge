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
def _default_s3_presign_to_sigv2(self, signature_version, **kwargs):
    """
        Returns the 's3' (sigv2) signer if presigning an s3 request. This is
        intended to be used to set the default signature version for the signer
        to sigv2. Situations where an asymmetric signature is required are the
        exception, for example MRAP needs v4a.

        :type signature_version: str
        :param signature_version: The current client signature version.

        :type signing_name: str
        :param signing_name: The signing name of the service.

        :return: 's3' if the request is an s3 presign request, None otherwise
        """
    if signature_version.startswith('v4a'):
        return
    if signature_version.startswith('v4-s3express'):
        return f'{signature_version}'
    for suffix in ['-query', '-presign-post']:
        if signature_version.endswith(suffix):
            return f's3{suffix}'