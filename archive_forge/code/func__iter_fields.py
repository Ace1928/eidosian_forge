import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def _iter_fields(self):
    _fields = self.fields
    if hasattr(self.fields, 'items'):
        _fields = list(self.fields.items())
    for k, v in _fields:
        file_name = None
        file_type = None
        file_headers = None
        if isinstance(v, (list, tuple)):
            if len(v) == 2:
                file_name, file_pointer = v
            elif len(v) == 3:
                file_name, file_pointer, file_type = v
            else:
                file_name, file_pointer, file_type, file_headers = v
        else:
            file_pointer = v
        field = fields.RequestField(name=k, data=file_pointer, filename=file_name, headers=file_headers)
        field.make_multipart(content_type=file_type)
        yield field