from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from torchgen import dest
from torchgen.api.types import DispatcherSignature  # isort:skip
from torchgen.context import method_with_native_function
from torchgen.executorch.model import ETKernelIndex
from torchgen.model import DispatchKey, NativeFunction, Variant
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, Target

    Generate custom ops registration code for dest.RegisterDispatchKey.

    :param native_functions: a sequence of `NativeFunction`
    :param selector: for selective build.
    :param kernel_index: kernels for all the ops.
    :param rocm: bool for dest.RegisterDispatchKey.
    :return: generated C++ code to register custom operators into PyTorch
    