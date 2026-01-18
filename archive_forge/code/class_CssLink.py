import base64
import json
import warnings
from binascii import hexlify
from collections import OrderedDict
from html import escape
from os import urandom
from pathlib import Path
from urllib.request import urlopen
from jinja2 import Environment, PackageLoader, Template
from .utilities import _camelify, _parse_size, none_max, none_min
class CssLink(Link):
    """Create a CssLink object based on a url.

    Parameters
    ----------
    url : str
        The url to be linked
    download : bool, default False
        Whether the target document shall be loaded right now.

    """
    _template = Template('{% if kwargs.get("embedded",False) %}<style>{{this.get_code()}}</style>{% else %}<link rel="stylesheet" href="{{this.url}}"/>{% endif %}')

    def __init__(self, url, download=False):
        super().__init__()
        self._name = 'CssLink'
        self.url = url
        self.code = None
        if download:
            self.get_code()