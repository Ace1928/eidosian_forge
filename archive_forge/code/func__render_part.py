from __future__ import absolute_import
import email.utils
import mimetypes
import re
from .packages import six
def _render_part(self, name, value):
    """
        Overridable helper function to format a single header parameter. By
        default, this calls ``self.header_formatter``.

        :param name:
            The name of the parameter, a string expected to be ASCII only.
        :param value:
            The value of the parameter, provided as a unicode string.
        """
    return self.header_formatter(name, value)