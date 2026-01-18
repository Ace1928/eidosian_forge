from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
def _needs_transcode(k):
    return _is_request(k) or k.startswith('HTTP_') or k.startswith('SSL_') or (k.startswith('REDIRECT_') and _needs_transcode(k[9:]))