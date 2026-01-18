import re
from itertools import zip_longest
from parso.python import tree
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, \
from jedi.inference.names import ParamName, TreeNameDefinition, AnonymousParamName
from jedi.inference.base_value import NO_VALUES, ValueSet, ContextualizedNode
from jedi.inference.value import iterable
from jedi.inference.cache import inference_state_as_method_param_cache
def _parse_argument_clinic(string):
    allow_kwargs = False
    optional = False
    while string:
        match = re.match('(?:(?:(\\[),? ?|, ?|)(\\**\\w+)|, ?/)\\]*', string)
        string = string[len(match.group(0)):]
        if not match.group(2):
            allow_kwargs = True
            continue
        optional = optional or bool(match.group(1))
        word = match.group(2)
        stars = word.count('*')
        word = word[stars:]
        yield (word, optional, allow_kwargs, stars)
        if stars:
            allow_kwargs = True