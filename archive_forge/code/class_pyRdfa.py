import sys
from io import StringIO, IOBase
import os
import xml.dom.minidom
from urllib.parse import urlparse
import rdflib
from rdflib import URIRef
from rdflib import Literal
from rdflib import BNode
from rdflib import Namespace
from rdflib import RDF as ns_rdf
from rdflib import RDFS as ns_rdfs
from rdflib import Graph
from .extras.httpheader import acceptable_content_type, content_type
from .transform.prototype import handle_prototypes
from .state import ExecutionContext
from .parse import parse_one_node
from .options import Options
from .transform import top_about, empty_safe_curie, vocab_for_role
from .utils import URIOpener
from .host import HostLanguage, MediaTypes, preferred_suffixes, content_to_host_language
class pyRdfa:
    """Main processing class for the distiller

    @ivar options: an instance of the L{Options} class
    @ivar media_type: the preferred default media type, possibly set at initialization
    @ivar base: the base value, possibly set at initialization
    @ivar http_status: HTTP Status, to be returned when the package is used via a CGI entry. Initially set to 200, may be modified by exception handlers
    """

    def __init__(self, options=None, base='', media_type='', rdfa_version=None):
        """
        @keyword options: Options for the distiller
        @type options: L{Options}
        @keyword base: URI for the default "base" value (usually the URI of the file to be processed)
        @keyword media_type: explicit setting of the preferred media type (a.k.a. content type) of the the RDFa source
        @keyword rdfa_version: the RDFa version that should be used. If not set, the value of the global L{rdfa_current_version} variable is used
        """
        self.http_status = 200
        self.base = base
        if base == '':
            self.required_base = None
        else:
            self.required_base = base
        self.charset = None
        self.media_type = media_type
        if options == None:
            self.options = Options()
        else:
            self.options = options
        if media_type != '':
            self.options.set_host_language(self.media_type)
        if rdfa_version is not None:
            self.rdfa_version = rdfa_version
        else:
            self.rdfa_version = None

    def _get_input(self, name):
        """
        Trying to guess whether "name" is a URI or a string (for a file); it then tries to open this source accordingly,
        returning a file-like object. If name is none of these, it returns the input argument (that should
        be, supposedly, a file-like object already).

        If the media type has not been set explicitly at initialization of this instance,
        the method also sets the media_type based on the HTTP GET response or the suffix of the file. See
        L{host.preferred_suffixes} for the suffix to media type mapping.

        @param name: identifier of the input source
        @type name: string or a file-like object
        @return: a file like object if opening "name" is possible and successful, "name" otherwise
        """
        isstring = isinstance(name, str)
        try:
            if isstring:
                if urlparse(name)[0] != '':
                    url_request = URIOpener(name, {}, self.options.certifi_verify)
                    self.base = url_request.location
                    if self.media_type == '':
                        if url_request.content_type in content_to_host_language:
                            self.media_type = url_request.content_type
                        else:
                            self.media_type = MediaTypes.xml
                        self.options.set_host_language(self.media_type)
                    self.charset = url_request.charset
                    if self.required_base == None:
                        self.required_base = name
                    return url_request.data
                else:
                    if self.required_base == None:
                        self.required_base = 'file://' + os.path.join(os.getcwd(), name)
                    if self.media_type == '':
                        self.media_type = MediaTypes.xml
                        for suffix in preferred_suffixes:
                            if name.endswith(suffix):
                                self.media_type = preferred_suffixes[suffix]
                                self.charset = 'utf-8'
                                break
                        self.options.set_host_language(self.media_type)
                    return open(name)
            else:
                return name
        except HTTPError:
            raise sys.exc_info()[1]
        except RDFaError as e:
            raise e
        except:
            _type, value, _traceback = sys.exc_info()
            raise FailedSource(value)

    @staticmethod
    def _validate_output_format(outputFormat):
        """
        Malicious actors may create XSS style issues by using an illegal output format... better be careful
        """
        if outputFormat not in ['turtle', 'n3', 'xml', 'pretty-xml', 'nt', 'json-ld']:
            outputFormat = 'turtle'
        return outputFormat

    def graph_from_DOM(self, dom, graph=None, pgraph=None):
        """
        Extract the RDF Graph from a DOM tree. This is where the real processing happens. All other methods get down to this
        one, eventually (e.g., after opening a URI and parsing it into a DOM).
        @param dom: a DOM Node element, the top level entry node for the whole tree (i.e., the C{dom.documentElement} is used to initiate processing down the node hierarchy)
        @keyword graph: an RDF Graph (if None, than a new one is created)
        @type graph: rdflib Graph instance.
        @keyword pgraph: an RDF Graph to hold (possibly) the processor graph content. If None, and the error/warning triples are to be generated, they will be added to the returned graph. Otherwise they are stored in this graph.
        @type pgraph: rdflib Graph instance
        @return: an RDF Graph
        @rtype: rdflib Graph instance
        """

        def copyGraph(tog, fromg):
            for t in fromg:
                tog.add(t)
            for k, ns in fromg.namespaces():
                tog.bind(k, ns)
        if graph == None:
            graph = Graph()
        default_graph = Graph()
        topElement = dom.documentElement
        state = ExecutionContext(topElement, default_graph, base=self.required_base if self.required_base != None else '', options=self.options, rdfa_version=self.rdfa_version)
        for trans in self.options.transformers + builtInTransformers:
            trans(topElement, self.options, state)
        self.rdfa_version = state.rdfa_version
        parse_one_node(topElement, default_graph, None, state, [])
        handle_prototypes(default_graph)
        if self.options.vocab_expansion:
            from .rdfs.process import process_rdfa_sem
            process_rdfa_sem(default_graph, self.options)
        if self.options.experimental_features:
            pass
        if self.options.output_default_graph:
            copyGraph(graph, default_graph)
            if self.options.output_processor_graph:
                if pgraph != None:
                    copyGraph(pgraph, self.options.processor_graph.graph)
                else:
                    copyGraph(graph, self.options.processor_graph.graph)
        elif self.options.output_processor_graph:
            if pgraph != None:
                copyGraph(pgraph, self.options.processor_graph.graph)
            else:
                copyGraph(graph, self.options.processor_graph.graph)
        self.options.reset_processor_graph()
        return graph

    def graph_from_source(self, name, graph=None, rdfOutput=False, pgraph=None):
        """
        Extract an RDF graph from an RDFa source. The source is parsed, the RDF extracted, and the RDFa Graph is
        returned. This is a front-end to the L{pyRdfa.graph_from_DOM} method.

        @param name: a URI, a file name, or a file-like object
        @param graph: rdflib Graph instance. If None, a new one is created.
        @param pgraph: rdflib Graph instance for the processor graph. If None, and the error/warning triples are to be generated, they will be added to the returned graph. Otherwise they are stored in this graph.
        @param rdfOutput: whether runtime exceptions should be turned into RDF and returned as part of the processor graph
        @return: an RDF Graph
        @rtype: rdflib Graph instance
        """

        def copyErrors(tog, options):
            if tog == None:
                tog = Graph()
            if options.output_processor_graph:
                for t in options.processor_graph.graph:
                    tog.add(t)
                    if pgraph != None:
                        pgraph.add(t)
                for k, ns in options.processor_graph.graph.namespaces():
                    tog.bind(k, ns)
                    if pgraph != None:
                        pgraph.bind(k, ns)
            options.reset_processor_graph()
            return tog
        isstring = isinstance(name, str)
        try:
            stream = None
            try:
                stream = self._get_input(name)
            except FailedSource as ex:
                _f = sys.exc_info()[1]
                self.http_status = 400
                if not rdfOutput:
                    raise Exception(ex.msg)
                err = self.options.add_error(ex.msg, FileReferenceError, name)
                self.options.processor_graph.add_http_context(err, 400)
                return copyErrors(graph, self.options)
            except HTTPError as ex:
                h = sys.exc_info()[1]
                self.http_status = h.http_code
                if not rdfOutput:
                    raise Exception(ex.msg)
                err = self.options.add_error('HTTP Error: %s (%s)' % (h.http_code, h.msg), HTError, name)
                self.options.processor_graph.add_http_context(err, h.http_code)
                return copyErrors(graph, self.options)
            except RDFaError as ex:
                e = sys.exc_info()[1]
                self.http_status = 500
                if not rdfOutput:
                    raise Exception(ex.msg)
                err = self.options.add_error(str(ex.msg), context=name)
                self.options.processor_graph.add_http_context(err, 500)
                return copyErrors(graph, self.options)
            except Exception as ex:
                e = sys.exc_info()[1]
                self.http_status = 500
                if not rdfOutput:
                    raise ex
                err = self.options.add_error(str(e), context=name)
                self.options.processor_graph.add_http_context(err, 500)
                return copyErrors(graph, self.options)
            dom = None
            try:
                msg = ''
                parser = None
                if self.options.host_language == HostLanguage.html5:
                    import warnings
                    warnings.filterwarnings('ignore', category=DeprecationWarning)
                    from html5lib import HTMLParser, treebuilders
                    parser = HTMLParser(tree=treebuilders.getTreeBuilder('dom'))
                    if self.charset:
                        dom = parser.parse(stream)
                    else:
                        dom = parser.parse(stream)
                    try:
                        if isstring:
                            stream.close()
                            stream = self._get_input(name)
                        else:
                            stream.seek(0)
                        from .host import adjust_html_version
                        self.rdfa_version = adjust_html_version(stream, self.rdfa_version)
                    except:
                        pass
                else:
                    from .host import adjust_xhtml_and_version
                    if isinstance(stream, IOBase):
                        parse = xml.dom.minidom.parse
                    else:
                        parse = xml.dom.minidom.parseString
                    dom = parse(stream)
                    adjusted_host_language, version = adjust_xhtml_and_version(dom, self.options.host_language, self.rdfa_version)
                    self.options.host_language = adjusted_host_language
                    self.rdfa_version = version
            except ImportError:
                msg = 'HTML5 parser not available. Try installing html5lib <http://code.google.com/p/html5lib>'
                raise ImportError(msg)
            except Exception:
                e = sys.exc_info()[1]
                err = self.options.add_error(str(e), context=name)
                self.http_status = 400
                self.options.processor_graph.add_http_context(err, 400)
                return copyErrors(graph, self.options)
            return self.graph_from_DOM(dom, graph, pgraph)
        except Exception:
            a, b, c = sys.exc_info()
            sys.excepthook(a, b, c)
            if isinstance(b, ImportError):
                self.http_status = None
            else:
                self.http_status = 500
            if not rdfOutput:
                raise b
            err = self.options.add_error(str(b), context=name)
            self.options.processor_graph.add_http_context(err, 500)
            return copyErrors(graph, self.options)

    def rdf_from_sources(self, names, outputFormat='turtle', rdfOutput=False):
        """
        Extract and RDF graph from a list of RDFa sources and serialize them in one graph. The sources are parsed, the RDF
        extracted, and serialization is done in the specified format.
        @param names: list of sources, each can be a URI, a file name, or a file-like object
        @keyword outputFormat: serialization format. Can be one of "turtle", "n3", "xml", "pretty-xml", "nt". "xml", "pretty-xml", "json" or "json-ld". "turtle" and "n3", "xml" and "pretty-xml", and "json" and "json-ld" are synonyms, respectively. Note that the JSON-LD serialization works with RDFLib 3.* only.
        @keyword rdfOutput: controls what happens in case an exception is raised. If the value is False, the caller is responsible handling it; otherwise a graph is returned with an error message included in the processor graph
        @type rdfOutput: boolean
        @return: a serialized RDF Graph
        @rtype: string
        """
        outputFormat = pyRdfa._validate_output_format(outputFormat)
        graph = Graph()
        for name in names:
            self.graph_from_source(name, graph, rdfOutput)
        return str(graph.serialize(format=outputFormat), encoding='utf-8')

    def rdf_from_source(self, name, outputFormat='turtle', rdfOutput=False):
        """
        Extract and RDF graph from an RDFa source and serialize it in one graph. The source is parsed, the RDF
        extracted, and serialization is done in the specified format.
        @param name: a URI, a file name, or a file-like object
        @keyword outputFormat: serialization format. Can be one of "turtle", "n3", "xml", "pretty-xml", "nt". "xml", "pretty-xml", or "json-ld". "turtle" and "n3", or "xml" and "pretty-xml" are synonyms, respectively. Note that the JSON-LD serialization works with RDFLib 3.* only.
        @keyword rdfOutput: controls what happens in case an exception is raised. If the value is False, the caller is responsible handling it; otherwise a graph is returned with an error message included in the processor graph
        @type rdfOutput: boolean
        @return: a serialized RDF Graph
        @rtype: string
        """
        return self.rdf_from_sources([name], outputFormat, rdfOutput)