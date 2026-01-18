import binascii
import unicodedata
import base64
import cherrypy
from cherrypy._cpcompat import ntou, tonative
def _try_decode(subject, charsets):
    for charset in charsets[:-1]:
        try:
            return tonative(subject, charset)
        except ValueError:
            pass
    return tonative(subject, charsets[-1])