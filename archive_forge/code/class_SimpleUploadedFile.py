import os
from io import BytesIO
from django.conf import settings
from django.core.files import temp as tempfile
from django.core.files.base import File
from django.core.files.utils import validate_file_name
class SimpleUploadedFile(InMemoryUploadedFile):
    """
    A simple representation of a file, which just has content, size, and a name.
    """

    def __init__(self, name, content, content_type='text/plain'):
        content = content or b''
        super().__init__(BytesIO(content), None, name, content_type, len(content), None, None)

    @classmethod
    def from_dict(cls, file_dict):
        """
        Create a SimpleUploadedFile object from a dictionary with keys:
           - filename
           - content-type
           - content
        """
        return cls(file_dict['filename'], file_dict['content'], file_dict.get('content-type', 'text/plain'))