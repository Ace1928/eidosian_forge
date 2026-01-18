import collections
import enum
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates
class _StateStack(object):
    """Templated context manager.

  This class provides syntactic sugar for a stack of objects of known
  type. It allows accessing attributes of the object at the top of the stack
  directly against this object, which allows for very terse syntax.

  For example, this code:

    stack = _StateStack(Foo)
    stack.enter()
    stack.bar

  Is equivalent to:

    stack = []
    stack.append(Foo())
    foo = stack[-1]
    foo.bar

  See _State for more on how this is used.

  Attributes:
    type: Any, the type of objects that this stack holds
    level: int, the current stack depth
    stack: List[Any], the actual stack
    value: Any, the instance of the object at the top of the stack
  """

    def __init__(self, type_):
        object.__setattr__(self, 'type', type_)
        object.__setattr__(self, '_stack', [])
        if not hasattr(type_, 'no_root'):
            self.enter()

    def __enter__(self):
        self.enter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()

    def enter(self):
        self._stack.append(self.type())

    def exit(self):
        self._stack.pop()

    @property
    def stack(self):
        return self._stack

    @property
    def level(self):
        return len(self._stack)

    @property
    def value(self):
        return self._stack[-1]

    def __iter__(self):
        return iter(self._stack)

    def __getattr__(self, key):
        return getattr(self._stack[-1], key)

    def __setattr__(self, key, value):
        setattr(self._stack[-1], key, value)