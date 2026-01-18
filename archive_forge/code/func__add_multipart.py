import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def _add_multipart(self, _subtype, *args, _disp=None, **kw):
    if self.get_content_maintype() != 'multipart' or self.get_content_subtype() != _subtype:
        getattr(self, 'make_' + _subtype)()
    part = type(self)(policy=self.policy)
    part.set_content(*args, **kw)
    if _disp and 'content-disposition' not in part:
        part['Content-Disposition'] = _disp
    self.attach(part)