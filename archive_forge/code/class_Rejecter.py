from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
class Rejecter(FilterCrutch):
    __slots__ = ()

    def __init__(self, builder):
        FilterCrutch.__init__(self, builder)
        parser = builder._parser
        for name in ('ProcessingInstructionHandler', 'CommentHandler', 'CharacterDataHandler', 'StartCdataSectionHandler', 'EndCdataSectionHandler', 'ExternalEntityRefHandler'):
            setattr(parser, name, None)

    def start_element_handler(self, *args):
        self._level = self._level + 1

    def end_element_handler(self, *args):
        if self._level == 0:
            parser = self._builder._parser
            self._builder.install(parser)
            parser.StartElementHandler = self._old_start
            parser.EndElementHandler = self._old_end
        else:
            self._level = self._level - 1