import re
from urllib.parse import urlsplit
from rdflib import URIRef
from rdflib import BNode
from rdflib import Namespace
from .utils import quote_URI
from .host import predefined_1_0_rel, warn_xmlns_usage
from . import IncorrectPrefixDefinition, RDFA_VOCAB, UnresolvableReference, PrefixRedefinitionWarning
from . import err_redefining_URI_as_prefix
from . import err_xmlns_deprecated
from . import err_bnode_local_prefix
from . import err_col_local_prefix
from . import err_missing_URI_prefix
from . import err_invalid_prefix
from . import err_no_default_prefix
from . import err_prefix_and_xmlns
from . import err_non_ncname_prefix
from . import err_absolute_reference
from . import err_query_reference
from . import err_fragment_reference
from . import err_prefix_redefinition
def CURIE_to_URI(self, val):
    """CURIE to URI mapping. 
        
        This method does I{not} take care of the last step of CURIE processing, ie, the fact that if
        it does not have a CURIE then the value is used a URI. This is done on the caller's side, because this has
        to be combined with base, for example. The method I{does} take care of BNode processing, though, ie,
        CURIE-s of the form "_:XXX".
        
        @param val: the full CURIE
        @type val: string
        @return: URIRef of a URI or None.
        """
    if val == '':
        return None
    elif val == ':':
        if self.default_curie_uri:
            return URIRef(self.default_curie_uri)
        else:
            return None
    curie_split = val.split(':', 1)
    if len(curie_split) == 1:
        return None
    else:
        if self.state.rdfa_version >= '1.1':
            prefix = curie_split[0].lower()
        else:
            prefix = curie_split[0]
        reference = curie_split[1]
        if len(prefix) == 0:
            if self.default_curie_uri and self._check_reference(reference):
                return self.default_curie_uri[reference]
            else:
                return None
        elif prefix == '_':
            if len(reference) == 0:
                return _empty_bnode
            elif reference in _bnodes:
                return _bnodes[reference]
            else:
                retval = BNode()
                _bnodes[reference] = retval
                return retval
        elif ncname.match(prefix):
            if prefix in self.ns and self._check_reference(reference):
                if len(reference) == 0:
                    return URIRef(str(self.ns[prefix]))
                else:
                    return self.ns[prefix][reference]
            elif prefix in self.default_prefixes and self._check_reference(reference):
                if len(reference) == 0:
                    return URIRef(str(self.default_prefixes[prefix][0]))
                else:
                    ns, used = self.default_prefixes[prefix]
                    if not used:
                        self.graph.bind(prefix, ns)
                        self.default_prefixes[prefix] = (ns, True)
                    return ns[reference]
            else:
                return None
        else:
            return None