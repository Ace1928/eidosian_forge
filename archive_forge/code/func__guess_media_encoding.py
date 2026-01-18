import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def _guess_media_encoding(self, source):
    info = source.byteStream.info()
    if 'Content-Type' in info:
        for param in info.getplist():
            if param.startswith('charset='):
                return param.split('=', 1)[1].lower()