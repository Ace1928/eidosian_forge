import collections
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
class Literal(collections.namedtuple('Literal', ['value'])):
    """Represents a Python numeric literal."""

    def __str__(self):
        if isinstance(self.value, str):
            return "'{}'".format(self.value)
        return str(self.value)

    def __repr__(self):
        return str(self)