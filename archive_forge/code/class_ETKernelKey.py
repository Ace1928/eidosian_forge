import itertools
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Union
from torchgen.model import (
from torchgen.utils import assert_never
@dataclass(frozen=True)
class ETKernelKey:
    arg_meta: Tuple[ETKernelKeyOpArgMeta, ...] = ()
    default: bool = False
    version: int = KERNEL_KEY_VERSION

    @staticmethod
    def gen_from_yaml(args: Dict[str, Tuple[str, str]], type_alias_map: Dict[str, List[str]], dim_order_alias_map: Dict[str, List[int]]) -> List['ETKernelKey']:
        """Generate ETKernelKeys from arg kernel specs
        Multiple ETKernelKeys are returned due to dtype permutations from utilizing
        type_alias_map (actualizing each potential type permutation as a KernelKey)

        Args:
            args: Mapping from argument name to kernel specs
                Kernel specs are a tuple of (dtype, dim_order).
                Currently tuple entries must be aliased via the alias map arguments
            type_alias_map: Mapping from type alias to potential type enums
                i.e { T0 : [Double, Int] } means T0 can be either Double or Int
                Used for lookup by args
            dim_order_alias_map: Mapping from alias to a list of dimension orders
                Used for lookup by args
        """
        dim_order_alias_map = {k: [int(alias) for alias in v] for k, v in dim_order_alias_map.items()}
        kernel_keys = []
        dtype_alias_used = set()
        for type_alias, dim_order in args.values():
            assert type_alias in type_alias_map, 'Undefined type alias: ' + str(type_alias)
            assert dim_order in dim_order_alias_map, 'Undefined dim_order alias: ' + str(dim_order)
            dtype_alias_used.add(type_alias)
        alias_dtypes = [[(alias, dtype) for dtype in type_alias_map[alias]] for alias in dtype_alias_used]
        alias_permutations = [dict(permutation) for permutation in list(itertools.product(*alias_dtypes))]
        op_arg_cache = {}
        for permutation in alias_permutations:
            arg_list = []
            for arg_name, arg_spec in args.items():
                dtype = permutation[arg_spec[0]]
                dim_order = dim_order_alias_map[arg_spec[1]]
                if (cache_key := (arg_name, dtype, tuple(dim_order))) not in op_arg_cache:
                    op_arg_cache[cache_key] = ETKernelKeyOpArgMeta(*cache_key)
                arg_list.append(op_arg_cache[cache_key])
            kernel_keys.append(ETKernelKey(tuple(arg_list)))
        return kernel_keys

    def to_native_string(self) -> str:
        if self.default:
            return 'default'
        return 'v' + str(KERNEL_KEY_VERSION) + '/' + '|'.join([arg.to_native_string() for arg in self.arg_meta])