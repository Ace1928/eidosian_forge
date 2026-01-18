from io import BytesIO
import struct
import random
import time
import dns.exception
import dns.tsig
from ._compat import long
def _set_section(self, section):
    """Set the renderer's current section.

        Sections must be rendered order: QUESTION, ANSWER, AUTHORITY,
        ADDITIONAL.  Sections may be empty.

        Raises dns.exception.FormError if an attempt was made to set
        a section value less than the current section.
        """
    if self.section != section:
        if self.section > section:
            raise dns.exception.FormError
        self.section = section