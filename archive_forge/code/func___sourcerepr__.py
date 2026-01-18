import copy
import types
from itertools import count
def __sourcerepr__(self, source, binding=None):
    if binding and len(self.__dict__) > 3:
        return self._source_repr_class(source, binding=binding)
    vals = self.__dict__.copy()
    if 'declarative_count' in vals:
        del vals['declarative_count']
    args = []
    if self.__unpackargs__ and self.__unpackargs__[0] == '*' and (self.__unpackargs__[1] in vals):
        v = vals[self.__unpackargs__[1]]
        if isinstance(v, (list, int)):
            args.extend(list(map(source.makeRepr, v)))
            del v[self.__unpackargs__[1]]
    for name in self.__unpackargs__:
        if name in vals:
            args.append(source.makeRepr(vals[name]))
            del vals[name]
        else:
            break
    args.extend(('%s=%s' % (name, source.makeRepr(value)) for name, value in vals.items()))
    return '%s(%s)' % (self.__class__.__name__, ', '.join(args))