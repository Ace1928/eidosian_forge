import re
import warnings
from parso import parse, ParserSyntaxError
from jedi import debug
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import iterator_to_value_set, ValueSet, \
from jedi.inference.lazy_value import LazyKnownValues
def _search_return_in_numpydocstr(docstr):
    """
    Search `docstr` (in numpydoc format) for type(-s) of function returns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            doc = _get_numpy_doc_string_cls()(docstr)
        except Exception:
            return
    try:
        returns = doc._parsed_data['Returns']
        returns += doc._parsed_data['Yields']
    except Exception:
        return
    for r_name, r_type, r_descr in returns:
        if not r_type:
            r_type = r_name
        yield from _expand_typestr(r_type)