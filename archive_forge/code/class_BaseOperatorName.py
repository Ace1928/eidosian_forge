import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class BaseOperatorName:
    base: str
    inplace: bool
    dunder_method: bool
    functional_overload: bool = False

    @staticmethod
    def parse(op: str) -> 'BaseOperatorName':
        assert op != ''
        assert not op.endswith('_out'), '_out suffix is reserved and not permitted for operator names; did you mean to specify an out overload name instead?'
        m = re.match('^__([^_]+)__$', op)
        if m is not None:
            dunder_method = True
            base = m.group(1)
            if any((base == f'i{n}' for n in AUGMENTED_ASSIGNMENT_NAMES)):
                inplace = True
                base = base[1:]
            else:
                inplace = False
                assert base[0] != 'i'
        else:
            dunder_method = False
            base = op
            if base[-1] == '_':
                inplace = True
                base = base[:-1]
            else:
                inplace = False
        functional_suffix = '_functional'
        if base.endswith(functional_suffix):
            functional_overload = True
            base = base[:-len(functional_suffix)]
            assert not dunder_method and (not inplace)
        else:
            functional_overload = False
        r = BaseOperatorName(base=base, inplace=inplace, dunder_method=dunder_method, functional_overload=functional_overload)
        assert str(r) == op, f'{str(r)} != {op}'
        return r

    def __str__(self) -> str:
        if self.dunder_method:
            i = 'i' if self.inplace else ''
            return f'__{i}{self.base}__'
        else:
            i = '_' if self.inplace else '_functional' if self.functional_overload else ''
            return f'{self.base}{i}'