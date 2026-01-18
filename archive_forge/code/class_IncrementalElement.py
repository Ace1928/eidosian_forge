import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
class IncrementalElement:
    """Wrapper for _IncrementalWriter providing an Element like interface.

    This wrapper does not intend to be a complete implementation but rather to
    deal with those calls used in GraphMLWriter.
    """

    def __init__(self, xml, prettyprint):
        self.xml = xml
        self.prettyprint = prettyprint

    def append(self, element):
        self.xml.write(element, pretty_print=self.prettyprint)