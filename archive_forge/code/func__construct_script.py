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
@staticmethod
def _construct_script(task_spec: common.TaskSpec, globals: GlobalsBridge, *, number: int, repeats: int, collect_baseline: bool, error_log: str, stat_log: str, bindings: Optional[CallgrindModuleType]) -> str:

    def block_stmt(stmt: str, indent: int=0) -> str:
        """Partially unroll benchmark loop.

            The naive template looks something like:
                "for _ in range({number}): {stmt}"

            However a loop in Python is surprisingly expensive, and significantly
            increases the number of background Python instructions. So instead we
            partially unroll the loops, with a block size of 100 chosen to keep
            the instruction overhead from `range` low while also not ballooning
            the size of the generated file.
            """
        block_size = 100
        loop_count = number // block_size
        if loop_count == 1:
            loop_count = 0
        remainder = number - block_size * loop_count
        blocked_stmt = ''
        if loop_count:
            unrolled_stmts = textwrap.indent('\n'.join([stmt] * block_size), ' ' * 4)
            blocked_stmt += f'for _ in range({loop_count}):\n{unrolled_stmts}\n'
        if remainder:
            blocked_stmt += '\n'.join([stmt] * remainder)
        return textwrap.indent(blocked_stmt, ' ' * indent)
    pass_baseline = f'callgrind_bindings._valgrind_toggle()\n{block_stmt('pass')}\ncallgrind_bindings._valgrind_toggle_and_dump_stats()'
    return textwrap.dedent('\n            import gc\n            import os\n            import pickle\n            import subprocess\n            import sys\n            import time\n\n            # Mitigate https://github.com/pytorch/pytorch/issues/37377\n            # which can sometimes cause the subprocess call to fail.\n            import numpy as np\n\n            import torch\n            torch.set_num_threads({num_threads})\n\n            {bindings_import}\n\n            PID = os.getpid()\n\n            def log_failure(msg):\n                with open({error_log_repr}, "wt") as f:\n                    f.write(msg)\n                sys.exit(1)\n\n            def check_result(completed_process):\n                if completed_process.returncode:\n                    log_failure(f"Command failed: {{\' \'.join(completed_process.args)}}")\n                return completed_process\n\n            # =============================================================================\n            # == Check that subprocess matches parent =====================================\n            # =============================================================================\n            if os.path.realpath(sys.executable) != "{parent_interpreter}":\n                log_failure(\n                    "Interpreter mismatch:\\n"\n                    f"  {{os.path.realpath(sys.executable)}}\\n    vs.\\n  {parent_interpreter}"\n                )\n\n            if torch.__file__ != "{torch_file}":\n                log_failure(\n                    "PyTorch does not match expected file:\\n"\n                    f"  {{torch.__file__}}\\n    vs.\\n  {torch_file}"\n                )\n\n            # =============================================================================\n            # == User specified setup =====================================================\n            # =============================================================================\n            # Load serialized globals\n            {load_globals}\n\n            # User setup str\n            {setup}\n\n            for _ in range({warmup_number}):\n            {indented_stmt}\n\n            # =============================================================================\n            # == Callgrind management =====================================================\n            # =============================================================================\n            with open("{stat_log}", "wb") as stat_file:\n                # If many instances of callgrind are running at once, the output of\n                # `callgrind_control` may exceed 16kb which would cause `subprocess.PIPE`\n                # to deadlock. So instead we use a file.\n                callgrind_stat = check_result(subprocess.run(\n                    ["callgrind_control", "--stat"],\n                    stdout=stat_file,\n                    stderr=subprocess.STDOUT,\n                ))\n\n            with open("{stat_log}", "rt") as stat_file:\n                stat_lines = stat_file.read().splitlines()\n\n            if f"PID {{PID}}: python {{__file__}}" not in stat_lines:\n                log_failure("Process does not appear to be running callgrind.")\n\n            gc.collect()\n            time.sleep(0.01)\n\n            # =============================================================================\n            # == User code block ==========================================================\n            # =============================================================================\n            for _ in range({repeats}):\n                callgrind_bindings._valgrind_toggle()\n            {blocked_stmt}\n                callgrind_bindings._valgrind_toggle_and_dump_stats()\n                gc.collect()\n\n            {baseline}\n        ').strip().format(indented_stmt=textwrap.indent(task_spec.stmt, ' ' * 4), blocked_stmt=block_stmt(task_spec.stmt, indent=4), baseline=pass_baseline if collect_baseline else '', number=number, repeats=repeats, load_globals=globals.construct(), setup=task_spec.setup, warmup_number=min(number, 10), num_threads=task_spec.num_threads, error_log_repr=repr(error_log), stat_log=stat_log, parent_interpreter=os.path.realpath(sys.executable), torch_file=torch.__file__, bindings_import='import torch._C as callgrind_bindings' if bindings is None else f'import {bindings.__name__} as callgrind_bindings')