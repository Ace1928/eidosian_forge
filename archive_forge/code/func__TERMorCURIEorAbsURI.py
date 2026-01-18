from rdflib import URIRef
from rdflib import BNode
from .host import HostLanguage, accept_xml_base, accept_xml_lang, beautifying_prefixes
from .termorcurie import TermOrCurie
from . import UnresolvablePrefix, UnresolvableTerm
from . import err_URI_scheme
from . import err_illegal_safe_CURIE
from . import err_no_CURIE_in_safe_CURIE
from . import err_undefined_terms
from . import err_non_legal_CURIE_ref
from . import err_undefined_CURIE
from urllib.parse import urlparse, urlunparse, urlsplit, urljoin
def _TERMorCURIEorAbsURI(self, val):
    """Returns a URI either for a term or for a CURIE. The value must be an NCNAME to be handled as a term; otherwise
        the method falls back on a CURIE or an absolute URI.
        @param val: attribute value to be interpreted
        @type val: string
        @return: an RDFLib URIRef instance or None
        """
    from . import uri_schemes
    if val == '':
        return None
    from .termorcurie import termname
    if termname.match(val):
        retval = self.term_or_curie.term_to_URI(val)
        if not retval:
            self.options.add_warning(err_undefined_terms % val, UnresolvableTerm, node=self.node.nodeName, buggy_value=val)
            return None
        else:
            return retval
    else:
        retval = self.term_or_curie.CURIE_to_URI(val)
        if retval:
            return retval
        elif self.rdfa_version >= '1.1':
            scheme = urlsplit(val)[0]
            if scheme == '':
                self.options.add_warning(err_non_legal_CURIE_ref % val, UnresolvablePrefix, node=self.node.nodeName)
                return None
            else:
                if scheme not in uri_schemes:
                    self.options.add_warning(err_URI_scheme % val.strip(), node=self.node.nodeName)
                return URIRef(val)
        else:
            self.options.add_warning(err_undefined_CURIE % val.strip(), UnresolvablePrefix, node=self.node.nodeName)
            return None