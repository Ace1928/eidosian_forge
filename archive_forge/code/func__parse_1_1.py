from .state import ExecutionContext
from .property import ProcessProperty
from .embeddedRDF import handle_embeddedRDF
from .host import HostLanguage, host_dom_transforms
from rdflib import URIRef
from rdflib import BNode
from rdflib import RDF as ns_rdf
from . import IncorrectBlankNodeUsage, err_no_blank_node
from .utils import has_one_of_attributes
def _parse_1_1(node, graph, parent_object, incoming_state, parent_incomplete_triples):
    """The (recursive) step of handling a single node. See the
    U{RDFa 1.1 Core document<http://www.w3.org/TR/rdfa-core/>} for further details.
    
    This is the RDFa 1.1 version.

    @param node: the DOM node to handle
    @param graph: the RDF graph
    @type graph: RDFLib's Graph object instance
    @param parent_object: the parent's object, as an RDFLib URIRef
    @param incoming_state: the inherited state (namespaces, lang, etc.)
    @type incoming_state: L{state.ExecutionContext}
    @param parent_incomplete_triples: list of hanging triples (the missing resource set to None) to be handled (or not)
    by the current node.
    @return: whether the caller has to complete it's parent's incomplete triples
    @rtype: Boolean
    """

    def header_check(p_obj):
        """Special disposition for the HTML <head> and <body> elements..."""
        if state.options.host_language in [HostLanguage.xhtml, HostLanguage.html5, HostLanguage.xhtml5]:
            if node.nodeName == 'head' or node.nodeName == 'body':
                if not has_one_of_attributes(node, 'about', 'resource', 'src', 'href'):
                    return p_obj
        else:
            return None

    def lite_check():
        if state.options.check_lite and state.options.host_language in [HostLanguage.html5, HostLanguage.xhtml5, HostLanguage.xhtml]:
            if node.tagName == 'link' and node.hasAttribute('rel') and (state.term_or_curie.CURIE_to_URI(node.getAttribute('rel')) != None):
                state.options.add_warning('In RDFa Lite, attribute @rel in <link> is only used in non-RDFa way (consider using @property)', node=node)
    state = ExecutionContext(node, graph, inherited_state=incoming_state)
    lite_check()
    handle_role_attribute(node, graph, state)
    if state.options.embedded_rdf and node.nodeType == node.ELEMENT_NODE and handle_embeddedRDF(node, graph, state):
        return
    if state.options.host_language in host_dom_transforms and node.nodeType == node.ELEMENT_NODE:
        for func in host_dom_transforms[state.options.host_language]:
            func(node, state)
    if not has_one_of_attributes(node, 'href', 'resource', 'about', 'property', 'rel', 'rev', 'typeof', 'src', 'vocab', 'prefix'):
        for n in node.childNodes:
            if n.nodeType == node.ELEMENT_NODE:
                parse_one_node(n, graph, parent_object, state, parent_incomplete_triples)
        return
    current_subject = None
    current_object = None
    typed_resource = None
    if has_one_of_attributes(node, 'rel', 'rev'):
        current_subject = header_check(parent_object)
        if node.hasAttribute('about'):
            current_subject = state.getURI('about')
            if node.hasAttribute('typeof'):
                typed_resource = current_subject
        if current_subject == None:
            current_subject = parent_object
        else:
            state.reset_list_mapping(origin=current_subject)
        current_object = state.getResource('resource', 'href', 'src')
        if node.hasAttribute('typeof') and (not node.hasAttribute('about')):
            if current_object == None:
                current_object = BNode()
            typed_resource = current_object
        if not node.hasAttribute('inlist') and current_object != None:
            state.reset_list_mapping(origin=current_object)
    elif node.hasAttribute('property') and (not has_one_of_attributes(node, 'content', 'datatype')):
        current_subject = header_check(parent_object)
        if node.hasAttribute('about'):
            current_subject = state.getURI('about')
            if node.hasAttribute('typeof'):
                typed_resource = current_subject
        if current_subject == None:
            current_subject = parent_object
        else:
            state.reset_list_mapping(origin=current_subject)
        if typed_resource == None and node.hasAttribute('typeof'):
            typed_resource = state.getResource('resource', 'href', 'src')
            if typed_resource == None:
                typed_resource = BNode()
            current_object = typed_resource
        else:
            current_object = current_subject
    else:
        current_subject = header_check(parent_object)
        if current_subject == None:
            current_subject = state.getResource('about', 'resource', 'href', 'src')
        if current_subject == None:
            if node.hasAttribute('typeof'):
                current_subject = BNode()
                state.reset_list_mapping(origin=current_subject)
            else:
                current_subject = parent_object
        else:
            state.reset_list_mapping(origin=current_subject)
        current_object = current_subject
        if node.hasAttribute('typeof'):
            typed_resource = current_subject
    for defined_type in state.getURI('typeof'):
        if typed_resource:
            graph.add((typed_resource, ns_rdf['type'], defined_type))
    incomplete_triples = []
    for prop in state.getURI('rel'):
        if not isinstance(prop, BNode):
            if node.hasAttribute('inlist'):
                if current_object != None:
                    state.add_to_list_mapping(prop, current_object)
                else:
                    state.add_to_list_mapping(prop, None)
                    incomplete_triples.append((None, prop, None))
            else:
                theTriple = (current_subject, prop, current_object)
                if current_object != None:
                    graph.add(theTriple)
                else:
                    incomplete_triples.append(theTriple)
        else:
            state.options.add_warning(err_no_blank_node % 'rel', warning_type=IncorrectBlankNodeUsage, node=node.nodeName)
    for prop in state.getURI('rev'):
        if not isinstance(prop, BNode):
            theTriple = (current_object, prop, current_subject)
            if current_object != None:
                graph.add(theTriple)
            else:
                incomplete_triples.append(theTriple)
        else:
            state.options.add_warning(err_no_blank_node % 'rev', warning_type=IncorrectBlankNodeUsage, node=node.nodeName)
    if node.hasAttribute('property'):
        ProcessProperty(node, graph, current_subject, state, typed_resource).generate_1_1()
    if current_object == None:
        object_to_children = BNode()
    else:
        object_to_children = current_object
    for n in node.childNodes:
        if n.nodeType == node.ELEMENT_NODE:
            _parse_1_1(n, graph, object_to_children, state, incomplete_triples)
    for s, p, o in parent_incomplete_triples:
        if s == None and o == None:
            incoming_state.add_to_list_mapping(p, current_subject)
        else:
            if s == None:
                s = current_subject
            if o == None:
                o = current_subject
            graph.add((s, p, o))
    if state.new_list and (not state.list_empty()):
        for prop in state.get_list_props():
            vals = state.get_list_value(prop)
            if vals == None:
                graph.add((state.get_list_origin(), prop, ns_rdf['nil']))
            else:
                heads = [BNode() for _r in vals] + [ns_rdf['nil']]
                for i in range(0, len(vals)):
                    graph.add((heads[i], ns_rdf['first'], vals[i]))
                    graph.add((heads[i], ns_rdf['rest'], heads[i + 1]))
                graph.add((state.get_list_origin(), prop, heads[0]))
    return