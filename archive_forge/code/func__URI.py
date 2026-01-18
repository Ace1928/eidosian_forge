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
def _URI(self, val):
    """Returns a URI for a 'pure' URI (ie, not a CURIE). The method resolves possible relative URI-s. It also
        checks whether the URI uses an unusual URI scheme (and issues a warning); this may be the result of an
        uninterpreted CURIE...
        @param val: attribute value to be interpreted
        @type val: string
        @return: an RDFLib URIRef instance
        """

    def create_URIRef(uri, check=True):
        """
            Mini helping function: it checks whether a uri is using a usual scheme before a URIRef is created. In case
            there is something unusual, a warning is generated (though the URIRef is created nevertheless)
            @param uri: (absolute) URI string
            @return: an RDFLib URIRef instance
            """
        from . import uri_schemes
        val = uri.strip()
        if check and urlsplit(val)[0] not in uri_schemes:
            self.options.add_warning(err_URI_scheme % val.strip(), node=self.node.nodeName)
        return URIRef(val)

    def join(base, v, check=True):
        """
            Mini helping function: it makes a urljoin for the paths. Based on the python library, but
            that one has a bug: in some cases it
            swallows the '#' or '?' character at the end. This is clearly a problem with
            Semantic Web URI-s, so this is checked, too
            @param base: base URI string
            @param v: local part
            @param check: whether the URI should be checked against the list of 'existing' URI schemes
            @return: an RDFLib URIRef instance
            """
        joined = urljoin(base, v)
        try:
            if v[-1] != joined[-1] and (v[-1] == '#' or v[-1] == '?'):
                return create_URIRef(joined + v[-1], check)
            else:
                return create_URIRef(joined, check)
        except:
            return create_URIRef(joined, check)
    if val == '':
        return URIRef(self.base)
    if self.parsedBase[0] == '':
        key = urlsplit(val)[0]
        if key == '':
            return join(self.base, val, check=False)
        else:
            return create_URIRef(val)
    else:
        return join(self.base, val)