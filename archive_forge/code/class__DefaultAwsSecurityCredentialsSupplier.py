import abc
from dataclasses import dataclass
import hashlib
import hmac
import http.client as http_client
import json
import os
import posixpath
import re
from typing import Optional
import urllib
from urllib.parse import urljoin
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
class _DefaultAwsSecurityCredentialsSupplier(AwsSecurityCredentialsSupplier):
    """Default implementation of AWS security credentials supplier. Supports retrieving
    credentials and region via EC2 metadata endpoints and environment variables.
    """

    def __init__(self, credential_source):
        self._region_url = credential_source.get('region_url')
        self._security_credentials_url = credential_source.get('url')
        self._imdsv2_session_token_url = credential_source.get('imdsv2_session_token_url')

    @_helpers.copy_docstring(AwsSecurityCredentialsSupplier)
    def get_aws_security_credentials(self, context, request):
        env_aws_access_key_id = os.environ.get(environment_vars.AWS_ACCESS_KEY_ID)
        env_aws_secret_access_key = os.environ.get(environment_vars.AWS_SECRET_ACCESS_KEY)
        env_aws_session_token = os.environ.get(environment_vars.AWS_SESSION_TOKEN)
        if env_aws_access_key_id and env_aws_secret_access_key:
            return AwsSecurityCredentials(env_aws_access_key_id, env_aws_secret_access_key, env_aws_session_token)
        imdsv2_session_token = self._get_imdsv2_session_token(request)
        role_name = self._get_metadata_role_name(request, imdsv2_session_token)
        credentials = self._get_metadata_security_credentials(request, role_name, imdsv2_session_token)
        return AwsSecurityCredentials(credentials.get('AccessKeyId'), credentials.get('SecretAccessKey'), credentials.get('Token'))

    @_helpers.copy_docstring(AwsSecurityCredentialsSupplier)
    def get_aws_region(self, context, request):
        env_aws_region = os.environ.get(environment_vars.AWS_REGION)
        if env_aws_region is not None:
            return env_aws_region
        env_aws_region = os.environ.get(environment_vars.AWS_DEFAULT_REGION)
        if env_aws_region is not None:
            return env_aws_region
        if not self._region_url:
            raise exceptions.RefreshError('Unable to determine AWS region')
        headers = None
        imdsv2_session_token = self._get_imdsv2_session_token(request)
        if imdsv2_session_token is not None:
            headers = {'X-aws-ec2-metadata-token': imdsv2_session_token}
        response = request(url=self._region_url, method='GET', headers=headers)
        response_body = response.data.decode('utf-8') if hasattr(response.data, 'decode') else response.data
        if response.status != http_client.OK:
            raise exceptions.RefreshError('Unable to retrieve AWS region: {}'.format(response_body))
        return response_body[:-1]

    def _get_imdsv2_session_token(self, request):
        if request is not None and self._imdsv2_session_token_url is not None:
            headers = {'X-aws-ec2-metadata-token-ttl-seconds': _IMDSV2_SESSION_TOKEN_TTL_SECONDS}
            imdsv2_session_token_response = request(url=self._imdsv2_session_token_url, method='PUT', headers=headers)
            if imdsv2_session_token_response.status != http_client.OK:
                raise exceptions.RefreshError('Unable to retrieve AWS Session Token: {}'.format(imdsv2_session_token_response.data))
            return imdsv2_session_token_response.data
        else:
            return None

    def _get_metadata_security_credentials(self, request, role_name, imdsv2_session_token):
        """Retrieves the AWS security credentials required for signing AWS
        requests from the AWS metadata server.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            role_name (str): The AWS role name required by the AWS metadata
                server security_credentials endpoint in order to return the
                credentials.
            imdsv2_session_token (str): The AWS IMDSv2 session token to be added as a
                header in the requests to AWS metadata endpoint.

        Returns:
            Mapping[str, str]: The AWS metadata server security credentials
                response.

        Raises:
            google.auth.exceptions.RefreshError: If an error occurs while
                retrieving the AWS security credentials.
        """
        headers = {'Content-Type': 'application/json'}
        if imdsv2_session_token is not None:
            headers['X-aws-ec2-metadata-token'] = imdsv2_session_token
        response = request(url='{}/{}'.format(self._security_credentials_url, role_name), method='GET', headers=headers)
        response_body = response.data.decode('utf-8') if hasattr(response.data, 'decode') else response.data
        if response.status != http_client.OK:
            raise exceptions.RefreshError('Unable to retrieve AWS security credentials: {}'.format(response_body))
        credentials_response = json.loads(response_body)
        return credentials_response

    def _get_metadata_role_name(self, request, imdsv2_session_token):
        """Retrieves the AWS role currently attached to the current AWS
        workload by querying the AWS metadata server. This is needed for the
        AWS metadata server security credentials endpoint in order to retrieve
        the AWS security credentials needed to sign requests to AWS APIs.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            imdsv2_session_token (str): The AWS IMDSv2 session token to be added as a
                header in the requests to AWS metadata endpoint.

        Returns:
            str: The AWS role name.

        Raises:
            google.auth.exceptions.RefreshError: If an error occurs while
                retrieving the AWS role name.
        """
        if self._security_credentials_url is None:
            raise exceptions.RefreshError('Unable to determine the AWS metadata server security credentials endpoint')
        headers = None
        if imdsv2_session_token is not None:
            headers = {'X-aws-ec2-metadata-token': imdsv2_session_token}
        response = request(url=self._security_credentials_url, method='GET', headers=headers)
        response_body = response.data.decode('utf-8') if hasattr(response.data, 'decode') else response.data
        if response.status != http_client.OK:
            raise exceptions.RefreshError('Unable to retrieve AWS role name {}'.format(response_body))
        return response_body