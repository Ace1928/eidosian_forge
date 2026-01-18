import rpy2.rlike.indexing as rli
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
class OrdDict(dict):
    """ Implements the Ordered Dict API defined in PEP 372.
    When `odict` becomes part of collections, this class
    should inherit from it rather than from `dict`.

    This class differs a little from the Ordered Dict
    proposed in PEP 372 by the fact that:
    not all elements have to be named. None as a key value means
    an absence of name for the element.

    """
    __l: List[Tuple[Optional[str], Any]]

    def __init__(self, c: Iterable[Tuple[Optional[str], Any]]=[]):
        if isinstance(c, TaggedList) or isinstance(c, OrdDict):
            c = c.items()
        elif isinstance(c, dict):
            raise TypeError('A regular dictionnary does not ' + 'conserve the order of its keys.')
        super(OrdDict, self).__init__()
        self.__l = []
        for k, v in c:
            self[k] = v

    def __copy__(self):
        cp = OrdDict(c=tuple(self.items()))
        return cp

    def __reduce__(self):
        return (self.__class__, (), {'_OrdDict__l': self.__l}, None, iter(self.items()))

    def __cmp__(self, o):
        return NotImplemented

    def __eq__(self, o):
        return NotImplemented

    def __getitem__(self, key: str):
        if key is None:
            raise ValueError('Unnamed items cannot be retrieved by key.')
        i = super(OrdDict, self).__getitem__(key)
        return self.__l[i][1]

    def __iter__(self):
        seq = self.__l
        for e in seq:
            k = e[0]
            if k is None:
                continue
            else:
                yield k

    def __len__(self):
        return len(self.__l)

    def __ne__(self, o):
        return NotImplemented

    def __repr__(self) -> str:
        s = ['o{']
        for k, v in self.items():
            s.append("'%s': %s, " % (str(k), str(v)))
        s.append('}')
        return ''.join(s)

    def __reversed__(self):
        raise NotImplementedError('Not yet implemented.')

    def __setitem__(self, key: Optional[str], value: Any):
        """ Replace the element if the key is known,
        and conserve its rank in the list, or append
        it if unknown. """
        if key is None:
            self.__l.append((key, value))
            return
        if key in self:
            i = self.index(key)
            self.__l[i] = (key, value)
        else:
            self.__l.append((key, value))
            super(OrdDict, self).__setitem__(key, len(self.__l) - 1)

    def byindex(self, i: int) -> Any:
        """ Fetch a value by index (rank), rather than by key."""
        return self.__l[i]

    def index(self, k: str) -> int:
        """ Return the index (rank) for the key 'k' """
        return super(OrdDict, self).__getitem__(k)

    def get(self, k: str, d: Any=None):
        """ OD.get(k[,d]) -> OD[k] if k in OD, else d.  d defaults to None """
        try:
            res = self[k]
        except KeyError:
            res = d
        return res

    def items(self):
        """ OD.items() -> an iterator over the (key, value) items of D """
        return iter(self.__l)

    def keys(self):
        """ """
        return tuple([x[0] for x in self.__l])

    def reverse(self):
        """ Reverse the order of the elements in-place (no copy)."""
        seq = self.__l
        n = len(self.__l)
        for i in range(n // 2):
            tmp = seq[i]
            seq[i] = seq[n - i - 1]
            kv = seq[i]
            if kv is not None:
                super(OrdDict, self).__setitem__(kv[0], i)
            seq[n - i - 1] = tmp
            kv = tmp
            if kv is not None:
                super(OrdDict, self).__setitem__(kv[0], n - i - 1)

    def sort(self, cmp=None, key=None, reverse=False):
        raise NotImplementedError('Not yet implemented.')