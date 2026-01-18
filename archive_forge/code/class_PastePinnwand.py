import errno
import subprocess
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse
import requests
import unicodedata
from .config import getpreferredencoding
from .translations import _
from ._typing_compat import Protocol
class PastePinnwand:

    def __init__(self, url: str, expiry: str) -> None:
        self.url = url
        self.expiry = expiry

    def paste(self, s: str) -> Tuple[str, str]:
        """Upload to pastebin via json interface."""
        url = urljoin(self.url, '/api/v1/paste')
        payload = {'expiry': self.expiry, 'files': [{'lexer': 'pycon', 'content': s}]}
        try:
            response = requests.post(url, json=payload, verify=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise PasteFailed(str(exc))
        data = response.json()
        paste_url = data['link']
        removal_url = data['removal']
        return (paste_url, removal_url)