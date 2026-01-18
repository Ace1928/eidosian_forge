import errno
import io
import os
import pathlib
from urllib.parse import urljoin
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.signals import setting_changed
from django.utils._os import safe_join
from django.utils.deconstruct import deconstructible
from django.utils.encoding import filepath_to_uri
from django.utils.functional import cached_property
from django.utils.timezone import now
from .base import Storage
from .mixins import StorageSettingsMixin
def _convert_stream_content(self, mode):
    """Convert actual file content according to the opening mode."""
    new_content_type = bytes if 'b' in mode else str
    if self._content_type == new_content_type:
        return
    content = self.file.getvalue()
    content = content.encode() if isinstance(content, str) else content.decode()
    self._content_type = new_content_type
    self._initialize_stream()
    self.file.write(content)