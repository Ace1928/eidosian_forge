from .xmlreader import InputSource
from .handler import ContentHandler, ErrorHandler
from ._exceptions import SAXException, SAXNotRecognizedException, \
import os, sys
def _create_parser(parser_name):
    drv_module = __import__(parser_name, {}, {}, ['create_parser'])
    return drv_module.create_parser()