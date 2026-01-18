from functools import partial
import email.base64mime
import email.quoprimime
from email import errors
from email.encoders import encode_7or8bit
def _get_encoder(self, header_bytes):
    if self.header_encoding == BASE64:
        return email.base64mime
    elif self.header_encoding == QP:
        return email.quoprimime
    elif self.header_encoding == SHORTEST:
        len64 = email.base64mime.header_length(header_bytes)
        lenqp = email.quoprimime.header_length(header_bytes)
        if len64 < lenqp:
            return email.base64mime
        else:
            return email.quoprimime
    else:
        return None