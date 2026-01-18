import re
import warnings
from parso import parse, ParserSyntaxError
from jedi import debug
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import iterator_to_value_set, ValueSet, \
from jedi.inference.lazy_value import LazyKnownValues
def _search_param_in_numpydocstr(docstr, param_str):
    """Search `docstr` (in numpydoc format) for type(-s) of `param_str`."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            params = _get_numpy_doc_string_cls()(docstr)._parsed_data['Parameters']
        except Exception:
            return []
    for p_name, p_type, p_descr in params:
        if p_name == param_str:
            m = re.match('([^,]+(,[^,]+)*?)(,[ ]*optional)?$', p_type)
            if m:
                p_type = m.group(1)
            return list(_expand_typestr(p_type))
    return []