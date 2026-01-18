import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _extract_stat_info(url, infile):
    """Extract the stat-like information from a DAV PROPFIND response.

    :param url: The url used for the PROPFIND request.
    :param infile: A file-like object pointing at the start of the response.
    """
    parser = xml.sax.make_parser()
    handler = DavStatHandler()
    handler.set_url(url)
    parser.setContentHandler(handler)
    infile.close = lambda: None
    try:
        parser.parse(infile)
    except xml.sax.SAXParseException as e:
        raise errors.InvalidHttpResponse(url, msg='Malformed xml response: %s' % e)
    if handler.is_dir:
        size = -1
        is_exec = True
    else:
        size = handler.length
        is_exec = handler.executable == 'T'
    return _DAVStat(size, handler.is_dir, is_exec)