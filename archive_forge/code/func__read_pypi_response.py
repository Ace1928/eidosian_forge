import os
from configparser import RawConfigParser
import warnings
from distutils.cmd import Command
def _read_pypi_response(self, response):
    """Read and decode a PyPI HTTP response."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        import cgi
    content_type = response.getheader('content-type', 'text/plain')
    encoding = cgi.parse_header(content_type)[1].get('charset', 'ascii')
    return response.read().decode(encoding)