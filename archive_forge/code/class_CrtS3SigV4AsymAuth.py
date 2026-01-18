import datetime
from io import BytesIO
from botocore.auth import (
from botocore.compat import HTTPHeaders, awscrt, parse_qs, urlsplit, urlunsplit
from botocore.exceptions import NoCredentialsError
from botocore.utils import percent_encode_sequence
class CrtS3SigV4AsymAuth(CrtSigV4AsymAuth):
    _USE_DOUBLE_URI_ENCODE = False
    _SHOULD_NORMALIZE_URI_PATH = False

    def _get_existing_sha256(self, request):
        return None

    def _should_sha256_sign_payload(self, request):
        client_config = request.context.get('client_config')
        s3_config = getattr(client_config, 's3', None)
        if s3_config is None:
            s3_config = {}
        sign_payload = s3_config.get('payload_signing_enabled', None)
        if sign_payload is not None:
            return sign_payload
        if not request.url.startswith('https') or 'Content-MD5' not in request.headers:
            return True
        if request.context.get('has_streaming_input', False):
            return False
        return super()._should_sha256_sign_payload(request)

    def _should_add_content_sha256_header(self, explicit_payload):
        return True