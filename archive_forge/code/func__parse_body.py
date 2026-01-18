import sys
import email.parser
from .encoder import encode_with
from requests.structures import CaseInsensitiveDict
def _parse_body(self, content):
    boundary = b''.join((b'--', self.boundary))

    def body_part(part):
        fixed = MultipartDecoder._fix_first_part(part, boundary)
        return BodyPart(fixed, self.encoding)

    def test_part(part):
        return part != b'' and part != b'\r\n' and (part[:4] != b'--\r\n') and (part != b'--')
    parts = content.split(b''.join((b'\r\n', boundary)))
    self.parts = tuple((body_part(x) for x in parts if test_part(x)))