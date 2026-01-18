import os
from io import BytesIO
from django.conf import settings
from django.core.files import temp as tempfile
from django.core.files.base import File
from django.core.files.utils import validate_file_name
class InMemoryUploadedFile(UploadedFile):
    """
    A file uploaded into memory (i.e. stream-to-memory).
    """

    def __init__(self, file, field_name, name, content_type, size, charset, content_type_extra=None):
        super().__init__(file, name, content_type, size, charset, content_type_extra)
        self.field_name = field_name

    def open(self, mode=None):
        self.file.seek(0)
        return self

    def chunks(self, chunk_size=None):
        self.file.seek(0)
        yield self.read()

    def multiple_chunks(self, chunk_size=None):
        return False