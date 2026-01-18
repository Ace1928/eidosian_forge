from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
class ImageCmsProfile:

    def __init__(self, profile):
        """
        :param profile: Either a string representing a filename,
            a file like object containing a profile or a
            low-level profile object

        """
        if isinstance(profile, str):
            if sys.platform == 'win32':
                profile_bytes_path = profile.encode()
                try:
                    profile_bytes_path.decode('ascii')
                except UnicodeDecodeError:
                    with open(profile, 'rb') as f:
                        self._set(core.profile_frombytes(f.read()))
                    return
            self._set(core.profile_open(profile), profile)
        elif hasattr(profile, 'read'):
            self._set(core.profile_frombytes(profile.read()))
        elif isinstance(profile, _imagingcms.CmsProfile):
            self._set(profile)
        else:
            msg = 'Invalid type for Profile'
            raise TypeError(msg)

    def _set(self, profile, filename=None):
        self.profile = profile
        self.filename = filename
        self.product_name = None
        self.product_info = None

    def tobytes(self):
        """
        Returns the profile in a format suitable for embedding in
        saved images.

        :returns: a bytes object containing the ICC profile.
        """
        return core.profile_tobytes(self.profile)