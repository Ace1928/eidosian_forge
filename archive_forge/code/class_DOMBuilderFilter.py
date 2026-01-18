import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
class DOMBuilderFilter:
    """Element filter which can be used to tailor construction of
    a DOM instance.
    """
    FILTER_ACCEPT = 1
    FILTER_REJECT = 2
    FILTER_SKIP = 3
    FILTER_INTERRUPT = 4
    whatToShow = NodeFilter.SHOW_ALL

    def _get_whatToShow(self):
        return self.whatToShow

    def acceptNode(self, element):
        return self.FILTER_ACCEPT

    def startContainer(self, element):
        return self.FILTER_ACCEPT