import io
import json
import os
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
def _get_url_data(self, request, url, headers):
    response = request(url=url, method='GET', headers=headers)
    response_body = response.data.decode('utf-8') if hasattr(response.data, 'decode') else response.data
    if response.status != 200:
        raise exceptions.RefreshError('Unable to retrieve Identity Pool subject token', response_body)
    return (response_body, url)