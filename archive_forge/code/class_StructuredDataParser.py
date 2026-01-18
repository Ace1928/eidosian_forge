from rdflib.parser import (
from . import pyRdfa, Options
from .embeddedRDF import handle_embeddedRDF
from .state import ExecutionContext
class StructuredDataParser(Parser):
    """
    Convenience parser to extract both RDFa (including embedded Turtle)
    and microdata from an HTML file.
    It is simply a wrapper around the specific parsers.
    """

    def parse(self, source, graph, pgraph=None, rdfa_version='', vocab_expansion=False, vocab_cache=False, media_type='text/html'):
        """
        @param source: one of the input sources that the RDFLib package defined
        @type source: InputSource class instance
        @param graph: target graph for the triples; output graph, in RDFa
        spec. parlance
        @keyword rdfa_version: 1.0 or 1.1. If the value is "", then, by
        default, 1.1 is used unless the source has explicit signals to use 1.0
        (e.g., using a @version attribute, using a DTD set up for 1.0, etc)
        @type rdfa_version: string
        @type graph: RDFLib Graph
        @keyword pgraph: target for error and warning triples; processor
        graph, in RDFa spec. parlance. If set to None, these triples are
        ignored
        @type pgraph: RDFLib Graph
        @keyword vocab_expansion: whether the RDFa @vocab attribute should
        also mean vocabulary expansion (see the RDFa 1.1 spec for further
            details)
        @type vocab_expansion: Boolean
        @keyword vocab_cache: in case vocab expansion is used, whether the
        expansion data (i.e., vocabulary) should be cached locally. This
        requires the ability for the local application to write on the
        local file system
        @type vocab_chache: Boolean
        @keyword rdfOutput: whether Exceptions should be catched and added,
        as triples, to the processor graph, or whether they should be raised.
        @type rdfOutput: Boolean
        """
        baseURI, orig_source = _get_orig_source(source)
        if rdfa_version == '':
            rdfa_version = '1.1'
        RDFaParser()._process(graph, pgraph, baseURI, orig_source, media_type='text/html', rdfa_version=rdfa_version, vocab_expansion=vocab_expansion, vocab_cache=vocab_cache)
        try:
            from pyMicrodata.rdflibparsers import MicrodataParser
            MicrodataParser()._process(graph, baseURI, orig_source)
        except ImportError:
            warnings.warn('pyMicrodata not installed, will only parse RDFa')
        HTurtleParser()._process(graph, baseURI, orig_source, media_type='text/html')