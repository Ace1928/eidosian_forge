import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
class DOMImplementationLS:
    MODE_SYNCHRONOUS = 1
    MODE_ASYNCHRONOUS = 2

    def createDOMBuilder(self, mode, schemaType):
        if schemaType is not None:
            raise xml.dom.NotSupportedErr('schemaType not yet supported')
        if mode == self.MODE_SYNCHRONOUS:
            return DOMBuilder()
        if mode == self.MODE_ASYNCHRONOUS:
            raise xml.dom.NotSupportedErr('asynchronous builders are not supported')
        raise ValueError('unknown value for mode')

    def createDOMWriter(self):
        raise NotImplementedError("the writer interface hasn't been written yet!")

    def createDOMInputSource(self):
        return DOMInputSource()