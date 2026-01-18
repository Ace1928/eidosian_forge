import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def _download_classifiers() -> str:
    import ssl
    from email.message import Message
    from urllib.request import urlopen
    url = 'https://pypi.org/pypi?:action=list_classifiers'
    context = ssl.create_default_context()
    with urlopen(url, context=context) as response:
        headers = Message()
        headers['content_type'] = response.getheader('content-type', 'text/plain')
        return response.read().decode(headers.get_param('charset', 'utf-8'))