import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def _extract_dir_content(url, infile):
    """Extract the directory content from a DAV PROPFIND response.

    :param url: The url used for the PROPFIND request.
    :param infile: A file-like object pointing at the start of the response.
    """
    parser = xml.sax.make_parser()
    handler = DavListDirHandler()
    handler.set_url(url)
    parser.setContentHandler(handler)
    infile.close = lambda: None
    try:
        parser.parse(infile)
    except xml.sax.SAXParseException as e:
        raise errors.InvalidHttpResponse(url, msg='Malformed xml response: %s' % e)
    dir_content = handler.dir_content
    dir_name, is_dir = dir_content[0][:2]
    if not is_dir:
        raise errors.NotADirectory(url)
    dir_len = len(dir_name)
    elements = []
    for href, is_dir, size, is_exec in dir_content[1:]:
        if href.startswith(dir_name):
            name = href[dir_len:]
            if name.endswith('/'):
                name = name[0:-1]
            elements.append((str(name), is_dir, size, is_exec))
    return elements