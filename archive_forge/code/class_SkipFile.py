import os
from io import BytesIO
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.utils.module_loading import import_string
class SkipFile(UploadFileException):
    """
    This exception is raised by an upload handler that wants to skip a given file.
    """
    pass