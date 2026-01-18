import re
import warnings
from parso import parse, ParserSyntaxError
from jedi import debug
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import iterator_to_value_set, ValueSet, \
from jedi.inference.lazy_value import LazyKnownValues
def _get_numpy_doc_string_cls():
    global _numpy_doc_string_cache
    if isinstance(_numpy_doc_string_cache, (ImportError, SyntaxError)):
        raise _numpy_doc_string_cache
    from numpydoc.docscrape import NumpyDocString
    _numpy_doc_string_cache = NumpyDocString
    return _numpy_doc_string_cache