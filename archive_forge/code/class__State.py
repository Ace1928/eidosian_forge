import collections
import enum
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates
class _State(object):
    """Syntactic sugar for accessing an instance of a StateStack context manager.

  This structure offers syntactic sugar over a dict of stacks of objects
  of known type. These structures are useful to keep state during AST walks.
  Multiple different scopes can be tracked in parallel. For example:

    s = _State()

    s[foo].enter()
    s[bar].enter()  # this will not affect s[foo]

  Element access has special semantics:
    * keys are a data type
    * element values are _StateStack(type=key) objects
    * missing elements are automatically added, similarly to defaultdict

  For example, the following block :

    _State s
    s[Foo]

  Is equivalent to:

    s = {}
    if Foo not in s:
      s[Foo] = Foo()
    s[Foo]

  See Base for how it's used.
  """

    def __init__(self):
        self._value = {}

    def __getitem__(self, key):
        if key not in self._value:
            self._value[key] = _StateStack(key)
        return self._value[key]