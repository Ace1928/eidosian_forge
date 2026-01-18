import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
class Collection(TypeOperator):

    def __init__(self, holder_type, key_type, value_type, iter_type):
        super(Collection, self).__init__('collection', [holder_type, key_type, value_type, iter_type])

    def __str__(self):
        t0 = prune(self.types[0])
        if isinstance(t0, TypeVariable):
            if isinstance(prune(self.types[1]), TypeVariable):
                return 'Iterable[{}]'.format(self.types[3])
            else:
                return 'Collection[{}, {}]'.format(self.types[1], self.types[2])
        if isinstance(t0, TypeOperator) and t0.name == 'traits':
            if all((isinstance(prune(t), TypeVariable) for t in t0.types)):
                return 'Collection[{}, {}]'.format(self.types[1], self.types[2])
            elif all((isinstance(prune(t), TypeVariable) for t in t0.types[:1] + t0.types[2:])):
                t01 = prune(t0.types[1])
                if isinstance(t01, TypeOperator) and t01.name == LenTrait.name:
                    return 'Sized'
        t00 = prune(t0.types[0])
        if isinstance(t00, TypeOperator):
            type_trait = t00.name
            if type_trait == 'list':
                return 'List[{}]'.format(self.types[2])
            if type_trait == 'set':
                return 'Set[{}]'.format(self.types[2])
            if type_trait == 'dict':
                return 'Dict[{}, {}]'.format(self.types[1], self.types[2])
            if type_trait == 'str':
                return 'str'
            if type_trait == 'file':
                return 'IO[str]'
            if type_trait == 'tuple':
                return 'Tuple[{}]'.format(', '.join(map(str, self.types[1:])))
            if type_trait == 'array':
                t01 = prune(t0.types[1])
                hasnolen = isinstance(t01, TypeOperator) and t01.name == NoLenTrait.name
                if hasnolen:
                    return str(self.types[2])

                def rec(n):
                    pn = prune(n)
                    if isinstance(pn, Collection):
                        traits = prune(pn.types[0])
                        if isinstance(traits, TypeVariable):
                            return (pn.types[3], 0)
                        len_trait = prune(traits.types[1])
                        haslen = isinstance(len_trait, TypeOperator) and len_trait.name == LenTrait.name
                        if haslen:
                            t, n = rec(pn.types[3])
                            return (t, n + 1)
                        else:
                            return (pn.types[2], 0)
                    else:
                        return (pn, 0)
                t, n = rec(self)
                if isinstance(t, TypeVariable):
                    return 'Array[{} d+, {}]'.format(n, t)
                else:
                    return 'Array[{}d, {}]'.format(n, t)
            if type_trait == 'gen':
                return 'Generator[{}]'.format(self.types[2])
        return super(Collection, self).__str__()