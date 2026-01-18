import abc
import json
import os
from typing import NamedTuple
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
class _FileSupplier(SubjectTokenSupplier):
    """ Internal implementation of subject token supplier which supports reading a subject token from a file."""

    def __init__(self, path, format_type, subject_token_field_name):
        self._path = path
        self._format_type = format_type
        self._subject_token_field_name = subject_token_field_name

    @_helpers.copy_docstring(SubjectTokenSupplier)
    def get_subject_token(self, context, request):
        if not os.path.exists(self._path):
            raise exceptions.RefreshError("File '{}' was not found.".format(self._path))
        with open(self._path, 'r', encoding='utf-8') as file_obj:
            token_content = _TokenContent(file_obj.read(), self._path)
        return _parse_token_data(token_content, self._format_type, self._subject_token_field_name)