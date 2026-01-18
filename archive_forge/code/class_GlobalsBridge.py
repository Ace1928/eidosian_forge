import collections
import enum
import dataclasses
import itertools as it
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
from typing import (
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import CallgrindModuleType
class GlobalsBridge:
    """Handle the transfer of (certain) globals when collecting Callgrind statistics.

    Key takeaway: Any globals passed must be wrapped in `CopyIfCallgrind` to
                  work with `Timer.collect_callgrind`.

    Consider the following code snippet:
    ```
        import pickle
        import timeit

        class Counter:
            value = 0

            def __call__(self):
                self.value += 1

        counter = Counter()
        timeit.Timer("counter()", globals={"counter": counter}).timeit(10)
        print(counter.value)  # 10

        timeit.Timer(
            "counter()",
            globals={"counter": pickle.loads(pickle.dumps(counter))}
        ).timeit(20)
        print(counter.value)  # Still 10
    ```

    In the first case, `stmt` is executed using the objects in `globals`;
    however, the addition of serialization and deserialization changes the
    semantics and may meaningfully change behavior.

    This is a practical consideration when collecting Callgrind statistics.
    Unlike `exec` based execution (which `timeit` uses under the hood) which
    can share in-memory data structures with the caller, Callgrind collection
    requires an entirely new process in order to run under Valgrind. This means
    that any data structures used for statement execution will have to be
    serialized and deserialized in the subprocess.

    In order to avoid surprising semantics from (user invisible) process
    boundaries, what can be passed through `globals` is severely restricted
    for `Timer.collect_callgrind`. It is expected that most setup should be
    achievable (albeit perhaps less ergonomically) by passing a `setup`
    string.

    There are, however, exceptions. One such class are TorchScripted functions.
    Because they require a concrete file with source code it is not possible
    to define them using a `setup` string. Another group are torch.nn.Modules,
    whose construction can be complex and prohibitively cumbersome to coerce
    into a `setup` string. Finally, most builtin types are sufficiently well
    behaved and sufficiently common to warrant allowing as well. (e.g.
    `globals={"n": 1}` is very convenient.)

    Fortunately, all have well defined serialization semantics. This class
    is responsible for enabling the Valgrind subprocess to use elements in
    `globals` so long as they are an allowed type.

    Caveats:
        The user is required to acknowledge this serialization by wrapping
        elements in `globals` with `CopyIfCallgrind`.

        While ScriptFunction and ScriptModule are expected to save and load
        quite robustly, it is up to the user to ensure that an nn.Module can
        un-pickle successfully.

        `torch.Tensor` and `np.ndarray` are deliberately excluded. The
        serialization/deserialization process perturbs the representation of a
        tensor in ways that could result in incorrect measurements. For example,
        if a tensor lives in pinned CPU memory, this fact would not be preserved
        by a dump, and that will in turn change the performance of certain CUDA
        operations.
    """

    def __init__(self, globals: Dict[str, Any], data_dir: str) -> None:
        self._globals: Dict[str, CopyIfCallgrind] = {}
        self._data_dir = data_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        if globals.get('torch', torch) is not torch:
            raise ValueError('`collect_callgrind` does not support mocking out `torch`.')
        for name, value in globals.items():
            if name in ('torch', '__builtins__'):
                continue
            if not isinstance(value, CopyIfCallgrind):
                raise ValueError('`collect_callgrind` requires that globals be wrapped in `CopyIfCallgrind` so that serialization is explicit.')
            self._globals[name] = value

    def construct(self) -> str:
        load_lines = []
        for name, wrapped_value in self._globals.items():
            if wrapped_value.setup is not None:
                load_lines.append(textwrap.dedent(wrapped_value.setup))
            if wrapped_value.serialization == Serialization.PICKLE:
                path = os.path.join(self._data_dir, f'{name}.pkl')
                load_lines.append(f"with open({repr(path)}, 'rb') as f:\n    {name} = pickle.load(f)")
                with open(path, 'wb') as f:
                    pickle.dump(wrapped_value.value, f)
            elif wrapped_value.serialization == Serialization.TORCH:
                path = os.path.join(self._data_dir, f'{name}.pt')
                load_lines.append(f'{name} = torch.load({repr(path)})')
                torch.save(wrapped_value.value, path)
            elif wrapped_value.serialization == Serialization.TORCH_JIT:
                path = os.path.join(self._data_dir, f'{name}.pt')
                load_lines.append(f'{name} = torch.jit.load({repr(path)})')
                with open(path, 'wb') as f:
                    torch.jit.save(wrapped_value.value, f)
            else:
                raise NotImplementedError(f'Unknown serialization method: {wrapped_value.serialization}')
        return '\n'.join(load_lines)