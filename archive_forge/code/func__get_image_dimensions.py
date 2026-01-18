import struct
import zlib
from django.core.files import File
def _get_image_dimensions(self):
    if not hasattr(self, '_dimensions_cache'):
        close = self.closed
        self.open()
        self._dimensions_cache = get_image_dimensions(self, close=close)
    return self._dimensions_cache