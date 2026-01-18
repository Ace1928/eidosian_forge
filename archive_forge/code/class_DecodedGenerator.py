import re
import sys
import time
import random
from copy import deepcopy
from io import StringIO, BytesIO
from email.utils import _has_surrogates
class DecodedGenerator(Generator):
    """Generates a text representation of a message.

    Like the Generator base class, except that non-text parts are substituted
    with a format string representing the part.
    """

    def __init__(self, outfp, mangle_from_=None, maxheaderlen=None, fmt=None, *, policy=None):
        """Like Generator.__init__() except that an additional optional
        argument is allowed.

        Walks through all subparts of a message.  If the subpart is of main
        type `text', then it prints the decoded payload of the subpart.

        Otherwise, fmt is a format string that is used instead of the message
        payload.  fmt is expanded with the following keywords (in
        %(keyword)s format):

        type       : Full MIME type of the non-text part
        maintype   : Main MIME type of the non-text part
        subtype    : Sub-MIME type of the non-text part
        filename   : Filename of the non-text part
        description: Description associated with the non-text part
        encoding   : Content transfer encoding of the non-text part

        The default value for fmt is None, meaning

        [Non-text (%(type)s) part of message omitted, filename %(filename)s]
        """
        Generator.__init__(self, outfp, mangle_from_, maxheaderlen, policy=policy)
        if fmt is None:
            self._fmt = _FMT
        else:
            self._fmt = fmt

    def _dispatch(self, msg):
        for part in msg.walk():
            maintype = part.get_content_maintype()
            if maintype == 'text':
                print(part.get_payload(decode=False), file=self)
            elif maintype == 'multipart':
                pass
            else:
                print(self._fmt % {'type': part.get_content_type(), 'maintype': part.get_content_maintype(), 'subtype': part.get_content_subtype(), 'filename': part.get_filename('[no filename]'), 'description': part.get('Content-Description', '[no description]'), 'encoding': part.get('Content-Transfer-Encoding', '[no encoding]')}, file=self)