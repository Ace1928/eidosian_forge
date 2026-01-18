import random
from jmespath import lexer
from jmespath.compat import with_repr_method
from jmespath import ast
from jmespath import exceptions
from jmespath import visitor
def _free_cache_entries(self):
    for key in random.sample(self._CACHE.keys(), int(self._MAX_SIZE / 2)):
        self._CACHE.pop(key, None)