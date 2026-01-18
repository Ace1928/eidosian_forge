import abc
import json
import os
from typing import NamedTuple
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
class _UrlSupplier(SubjectTokenSupplier):
    """ Internal implementation of subject token supplier which supports retrieving a subject token by calling a URL endpoint."""

    def __init__(self, url, format_type, subject_token_field_name, headers):
        self._url = url
        self._format_type = format_type
        self._subject_token_field_name = subject_token_field_name
        self._headers = headers

    @_helpers.copy_docstring(SubjectTokenSupplier)
    def get_subject_token(self, context, request):
        response = request(url=self._url, method='GET', headers=self._headers)
        response_body = response.data.decode('utf-8') if hasattr(response.data, 'decode') else response.data
        if response.status != 200:
            raise exceptions.RefreshError('Unable to retrieve Identity Pool subject token', response_body)
        token_content = _TokenContent(response_body, self._url)
        return _parse_token_data(token_content, self._format_type, self._subject_token_field_name)