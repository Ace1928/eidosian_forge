import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class Precompute:
    replace: Dict[str, List[Argument]]
    add: List[Argument]

    @staticmethod
    def parse(src: object) -> 'Precompute':
        assert isinstance(src, list)
        add_args = []
        if ' -> ' not in src[-1]:
            add_list = src[-1].split(',')
            add_args = [Argument.parse(name.strip()) for name in add_list]
            src = src[:-1]
        replace = {}
        for raw_replace_item in src:
            assert isinstance(raw_replace_item, str)
            assert ' -> ' in raw_replace_item, 'precomputed parameters without replacement are allowed only in the last line'
            arg, with_list_raw = raw_replace_item.split(' -> ')
            with_list = with_list_raw.split(',')
            with_list_args = [Argument.parse(name.strip()) for name in with_list]
            replace[arg] = with_list_args
        r = Precompute(replace=replace, add=add_args)
        assert r.to_list() == src, 'r.to_list() != src'
        return r

    def __post_init__(self) -> None:
        for a in self.add:
            assert a.name.upper() != a.name
        for args in self.replace.values():
            for a in args:
                assert a.name.upper() != a.name

    def to_list(self) -> List[str]:
        replace_list = []
        for kernel_param, replacement_params in self.replace.items():
            replacements = ', '.join((str(param) for param in replacement_params))
            replace_list.append(f'{kernel_param} -> {replacements}')
        return replace_list