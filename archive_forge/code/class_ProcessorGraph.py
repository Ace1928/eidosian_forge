import datetime
from rdflib import URIRef
from rdflib import Literal
from rdflib import BNode
from rdflib import Namespace
from rdflib import Graph
from rdflib import RDF as ns_rdf
from .host import HostLanguage, content_to_host_language, predefined_1_0_rel, require_embedded_rdf
from . import ns_xsd, ns_distill, ns_rdfa
from . import RDFA_Error, RDFA_Warning, RDFA_Info
from .transform.lite import lite_prune
class ProcessorGraph:
    """Wrapper around the 'processor graph', ie, the (RDF) Graph containing the warnings,
    error messages, and informational messages.
    """

    def __init__(self):
        self.graph = Graph()

    def add_triples(self, msg, top_class, info_class, context, node):
        """
        Add an error structure to the processor graph: a bnode with a number of predicates. The structure
        follows U{the processor graph vocabulary<http://www.w3.org/2010/02/rdfa/wiki/Processor_Graph_Vocabulary>} as described
        on the RDFa WG Wiki page.
        
        @param msg: the core error message, added as an object to a dc:description
        @param top_class: Error, Warning, or Info; an explicit rdf:type added to the bnode
        @type top_class: URIRef
        @param info_class: An additional error class, added as an rdf:type to the bnode in case it is not None
        @type info_class: URIRef
        @param context: An additional information added, if not None, as an object with rdfa:context as a predicate
        @type context: either an URIRef or a URI String (an URIRef will be created in the second case)
        @param node: The node's element name that contains the error
        @type node: string
        @return: the bnode that serves as a subject for the errors. The caller may add additional information
        @rtype: BNode
        """
        self.graph.bind('dcterms', ns_dc)
        self.graph.bind('pyrdfa', ns_distill)
        self.graph.bind('rdf', ns_rdf)
        self.graph.bind('rdfa', ns_rdfa)
        self.graph.bind('ht', ns_ht)
        self.graph.bind('xsd', ns_xsd)
        is_context_string = isinstance(context, str)
        bnode = BNode()
        if node != None:
            try:
                full_msg = "[In element '%s'] %s" % (node.nodeName, msg)
            except:
                full_msg = "[In element '%s'] %s" % (node, msg)
        else:
            full_msg = msg
        self.graph.add((bnode, ns_rdf['type'], top_class))
        if info_class:
            self.graph.add((bnode, ns_rdf['type'], info_class))
        self.graph.add((bnode, ns_dc['description'], Literal(full_msg)))
        self.graph.add((bnode, ns_dc['date'], Literal(datetime.datetime.utcnow().isoformat(), datatype=ns_xsd['dateTime'])))
        if context and (isinstance(context, URIRef) or is_context_string):
            htbnode = BNode()
            self.graph.add((bnode, ns_rdfa['context'], htbnode))
            self.graph.add((htbnode, ns_rdf['type'], ns_ht['Request']))
            self.graph.add((htbnode, ns_ht['requestURI'], Literal('%s' % context)))
        return bnode

    def add_http_context(self, subj, http_code):
        """
        Add an additional HTTP context to a message with subject in C{subj}, using the U{<http://www.w3.org/2006/http#>}
        vocabulary. Typically used to extend an error structure, as created by L{add_triples}.
        
        @param subj: an RDFLib resource, typically a blank node
        @param http_code: HTTP status code
        """
        bnode = BNode()
        self.graph.add((subj, ns_rdfa['context'], bnode))
        self.graph.add((bnode, ns_rdf['type'], ns_ht['Response']))
        self.graph.add((bnode, ns_ht['responseCode'], URIRef('http://www.w3.org/2006/http#%s' % http_code)))