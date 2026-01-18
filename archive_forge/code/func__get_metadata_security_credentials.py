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
        raise exceptions.RefreshError('Unable to retrieve AWS security credentials', response_body)
    credentials_response = json.loads(response_body)
    return credentials_response