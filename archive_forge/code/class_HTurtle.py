from rdflib.parser import (
from . import pyRdfa, Options
from .embeddedRDF import handle_embeddedRDF
from .state import ExecutionContext
class HTurtle(pyRdfa):
    """
    Bastardizing the RDFa 1.1 parser to do a hturtle extractions
    """

    def __init__(self, options=None, base='', media_type=''):
        pyRdfa.__init__(self, options=options, base=base, media_type=media_type, rdfa_version='1.1')

    def graph_from_DOM(self, dom, graph, pgraph=None):
        """
        Stealing the parsing function from the original class, to do
        turtle extraction only
        """

        def copyGraph(tog, fromg):
            for t in fromg:
                tog.add(t)
            for k, ns in fromg.namespaces():
                tog.bind(k, ns)

        def _process_one_node(node, graph, state):
            if handle_embeddedRDF(node, graph, state):
                return
            else:
                for n in node.childNodes:
                    if n.nodeType == node.ELEMENT_NODE:
                        _process_one_node(n, graph, state)
        topElement = dom.documentElement
        state = ExecutionContext(topElement, graph, base=self.base, options=self.options, rdfa_version='1.1')
        _process_one_node(topElement, graph, state)
        if pgraph is not None:
            copyGraph(pgraph, self.options.processor_graph.graph)