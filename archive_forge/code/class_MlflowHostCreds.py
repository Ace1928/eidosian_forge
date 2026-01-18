import base64
import json
import requests
from mlflow.environment_variables import (
from mlflow.exceptions import (
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import ENDPOINT_NOT_FOUND, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.utils.proto_json_utils import parse_dict
from mlflow.utils.request_utils import (
from mlflow.utils.string_utils import strip_suffix
class MlflowHostCreds:
    """
    Provides a hostname and optional authentication for talking to an MLflow tracking server.

    Args:
        host: Hostname (e.g., http://localhost:5000) to MLflow server. Required.
        username: Username to use with Basic authentication when talking to server.
            If this is specified, password must also be specified.
        password: Password to use with Basic authentication when talking to server.
            If this is specified, username must also be specified.
        token: Token to use with Bearer authentication when talking to server.
            If provided, user/password authentication will be ignored.
        aws_sigv4: If true, we will create a signature V4 to be added for any outgoing request.
            Keys for signing the request can be passed via ENV variables,
            or will be fetched via boto3 session.
        auth: If set, the auth will be added for any outgoing request.
            Keys for signing the request can be passed via ENV variables,
        ignore_tls_verification: If true, we will not verify the server's hostname or TLS
            certificate. This is useful for certain testing situations, but should never be
            true in production.
            If this is set to true ``server_cert_path`` must not be set.
        client_cert_path: Path to ssl client cert file (.pem).
            Sets the cert param of the ``requests.request``
            function (see https://requests.readthedocs.io/en/master/api/).
        server_cert_path: Path to a CA bundle to use.
            Sets the verify param of the ``requests.request``
            function (see https://requests.readthedocs.io/en/master/api/).
            If this is set ``ignore_tls_verification`` must be false.
    """

    def __init__(self, host, username=None, password=None, token=None, aws_sigv4=False, auth=None, ignore_tls_verification=False, client_cert_path=None, server_cert_path=None):
        if not host:
            raise MlflowException(message='host is a required parameter for MlflowHostCreds', error_code=INVALID_PARAMETER_VALUE)
        if ignore_tls_verification and server_cert_path is not None:
            raise MlflowException(message="When 'ignore_tls_verification' is true then 'server_cert_path' must not be set! This error may have occurred because the 'MLFLOW_TRACKING_INSECURE_TLS' and 'MLFLOW_TRACKING_SERVER_CERT_PATH' environment variables are both set - only one of these environment variables may be set.", error_code=INVALID_PARAMETER_VALUE)
        self.host = host
        self.username = username
        self.password = password
        self.token = token
        self.aws_sigv4 = aws_sigv4
        self.auth = auth
        self.ignore_tls_verification = ignore_tls_verification
        self.client_cert_path = client_cert_path
        self.server_cert_path = server_cert_path

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    @property
    def verify(self):
        if self.server_cert_path is None:
            return not self.ignore_tls_verification
        else:
            return self.server_cert_path