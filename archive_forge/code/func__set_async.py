import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def _set_async(self, flag):
    if flag:
        raise xml.dom.NotSupportedErr('asynchronous document loading is not supported')