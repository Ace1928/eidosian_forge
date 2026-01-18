import hashlib
import hmac
import json
import os
import posixpath
import re
from six.moves import http_client
from six.moves import urllib
from six.moves.urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
class RequestSigner(object):
    """Implements an AWS request signer based on the AWS Signature Version 4 signing
    process.
    https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html
    """

    def __init__(self, region_name):
        """Instantiates an AWS request signer used to compute authenticated signed
        requests to AWS APIs based on the AWS Signature Version 4 signing process.

        Args:
            region_name (str): The AWS region to use.
        """
        self._region_name = region_name

    def get_request_options(self, aws_security_credentials, url, method, request_payload='', additional_headers={}):
        """Generates the signed request for the provided HTTP request for calling
        an AWS API. This follows the steps described at:
        https://docs.aws.amazon.com/general/latest/gr/sigv4_signing.html

        Args:
            aws_security_credentials (Mapping[str, str]): A dictionary containing
                the AWS security credentials.
            url (str): The AWS service URL containing the canonical URI and
                query string.
            method (str): The HTTP method used to call this API.
            request_payload (Optional[str]): The optional request payload if
                available.
            additional_headers (Optional[Mapping[str, str]]): The optional
                additional headers needed for the requested AWS API.

        Returns:
            Mapping[str, str]: The AWS signed request dictionary object.
        """
        access_key = aws_security_credentials.get('access_key_id')
        secret_key = aws_security_credentials.get('secret_access_key')
        security_token = aws_security_credentials.get('security_token')
        additional_headers = additional_headers or {}
        uri = urllib.parse.urlparse(url)
        normalized_uri = urllib.parse.urlparse(urljoin(url, posixpath.normpath(uri.path)))
        if not uri.hostname or uri.scheme != 'https':
            raise exceptions.InvalidResource('Invalid AWS service URL')
        header_map = _generate_authentication_header_map(host=uri.hostname, canonical_uri=normalized_uri.path or '/', canonical_querystring=_get_canonical_querystring(uri.query), method=method, region=self._region_name, access_key=access_key, secret_key=secret_key, security_token=security_token, request_payload=request_payload, additional_headers=additional_headers)
        headers = {'Authorization': header_map.get('authorization_header'), 'host': uri.hostname}
        if 'amz_date' in header_map:
            headers[_AWS_DATE_HEADER] = header_map.get('amz_date')
        for key in additional_headers:
            headers[key] = additional_headers[key]
        if security_token is not None:
            headers[_AWS_SECURITY_TOKEN_HEADER] = security_token
        signed_request = {'url': url, 'method': method, 'headers': headers}
        if request_payload:
            signed_request['data'] = request_payload
        return signed_request