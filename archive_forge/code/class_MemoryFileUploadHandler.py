import os
from io import BytesIO
from django.conf import settings
from django.core.files.uploadedfile import InMemoryUploadedFile, TemporaryUploadedFile
from django.utils.module_loading import import_string
class MemoryFileUploadHandler(FileUploadHandler):
    """
    File upload handler to stream uploads into memory (used for small files).
    """

    def handle_raw_input(self, input_data, META, content_length, boundary, encoding=None):
        """
        Use the content_length to signal whether or not this handler should be
        used.
        """
        self.activated = content_length <= settings.FILE_UPLOAD_MAX_MEMORY_SIZE

    def new_file(self, *args, **kwargs):
        super().new_file(*args, **kwargs)
        if self.activated:
            self.file = BytesIO()
            raise StopFutureHandlers()

    def receive_data_chunk(self, raw_data, start):
        """Add the data to the BytesIO file."""
        if self.activated:
            self.file.write(raw_data)
        else:
            return raw_data

    def file_complete(self, file_size):
        """Return a file object if this handler is activated."""
        if not self.activated:
            return
        self.file.seek(0)
        return InMemoryUploadedFile(file=self.file, field_name=self.field_name, name=self.file_name, content_type=self.content_type, size=file_size, charset=self.charset, content_type_extra=self.content_type_extra)