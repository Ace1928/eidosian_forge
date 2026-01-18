import os
from io import BytesIO
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.utils.module_loading import import_string
class FileUploadHandler:
    """
    Base class for streaming upload handlers.
    """
    chunk_size = 64 * 2 ** 10

    def __init__(self, request=None):
        self.file_name = None
        self.content_type = None
        self.content_length = None
        self.charset = None
        self.content_type_extra = None
        self.request = request

    def handle_raw_input(self, input_data, META, content_length, boundary, encoding=None):
        """
        Handle the raw input from the client.

        Parameters:

            :input_data:
                An object that supports reading via .read().
            :META:
                ``request.META``.
            :content_length:
                The (integer) value of the Content-Length header from the
                client.
            :boundary: The boundary from the Content-Type header. Be sure to
                prepend two '--'.
        """
        pass

    def new_file(self, field_name, file_name, content_type, content_length, charset=None, content_type_extra=None):
        """
        Signal that a new file has been started.

        Warning: As with any data from the client, you should not trust
        content_length (and sometimes won't even get it).
        """
        self.field_name = field_name
        self.file_name = file_name
        self.content_type = content_type
        self.content_length = content_length
        self.charset = charset
        self.content_type_extra = content_type_extra

    def receive_data_chunk(self, raw_data, start):
        """
        Receive data from the streamed upload parser. ``start`` is the position
        in the file of the chunk.
        """
        raise NotImplementedError('subclasses of FileUploadHandler must provide a receive_data_chunk() method')

    def file_complete(self, file_size):
        """
        Signal that a file has completed. File size corresponds to the actual
        size accumulated by all the chunks.

        Subclasses should return a valid ``UploadedFile`` object.
        """
        raise NotImplementedError('subclasses of FileUploadHandler must provide a file_complete() method')

    def upload_complete(self):
        """
        Signal that the upload is complete. Subclasses should perform cleanup
        that is necessary for this handler.
        """
        pass

    def upload_interrupted(self):
        """
        Signal that the upload was interrupted. Subclasses should perform
        cleanup that is necessary for this handler.
        """
        pass