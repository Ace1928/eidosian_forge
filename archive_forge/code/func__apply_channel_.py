import itertools
from collections import defaultdict
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import linalg, ops, protocols, value
from cirq.linalg import transformations
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.synchronize_terminal_measurements import find_terminal_measurements
def _apply_channel_(self, args: 'cirq.ApplyChannelArgs'):
    configs: List[transformations._BuildFromSlicesArgs] = []
    for i in range(np.prod(self._shape) ** 2):
        scale = cast(complex, self._confusion_map.flat[i])
        if scale == 0:
            continue
        index: Any = np.unravel_index(i, self._shape * 2)
        slices: List[transformations._SliceConfig] = []
        axis_count = len(args.left_axes)
        for j in range(axis_count):
            s1 = transformations._SliceConfig(axis=args.left_axes[j], source_index=index[j], target_index=index[j + axis_count])
            s2 = transformations._SliceConfig(axis=args.right_axes[j], source_index=index[j], target_index=index[j + axis_count])
            slices.extend([s1, s2])
        configs.append(transformations._BuildFromSlicesArgs(slices=tuple(slices), scale=scale))
    transformations._build_from_slices(configs, args.target_tensor, out=args.out_buffer)
    return args.out_buffer