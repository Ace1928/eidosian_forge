import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def _create_opener(self):
    import urllib.request
    return urllib.request.build_opener()