import weakref
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import transformer
class _NodeState(object):
    """Abstraction for the state of the CFG walk for reaching definition analysis.

  This is a value type. Only implements the strictly necessary operators.

  Attributes:
    value: Dict[qual_names.QN, Set[Definition, ...]], the defined symbols and
        their possible definitions
  """

    def __init__(self, init_from=None):
        if init_from:
            if isinstance(init_from, _NodeState):
                self.value = {s: set(other_infos) for s, other_infos in init_from.value.items()}
            elif isinstance(init_from, dict):
                self.value = {s: set((init_from[s],)) for s in init_from}
            else:
                assert False, init_from
        else:
            self.value = {}

    def __eq__(self, other):
        if frozenset(self.value.keys()) != frozenset(other.value.keys()):
            return False
        ret = all((self.value[s] == other.value[s] for s in self.value))
        return ret

    def __ne__(self, other):
        return not self.__eq__(other)

    def __or__(self, other):
        assert isinstance(other, _NodeState)
        result = _NodeState(self)
        for s, other_infos in other.value.items():
            if s in result.value:
                result.value[s].update(other_infos)
            else:
                result.value[s] = set(other_infos)
        return result

    def __sub__(self, other):
        assert isinstance(other, set)
        result = _NodeState(self)
        for s in other:
            result.value.pop(s, None)
        return result

    def __repr__(self):
        return 'NodeState[%s]=%s' % (id(self), repr(self.value))