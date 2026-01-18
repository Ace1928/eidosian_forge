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
class _ValgrindWrapper:

    def __init__(self) -> None:
        self._bindings_module: Optional[CallgrindModuleType] = None
        valgrind_symbols = ('_valgrind_supported_platform', '_valgrind_toggle', '_valgrind_toggle_and_dump_stats')
        if all((hasattr(torch._C, symbol) for symbol in valgrind_symbols)):
            self._supported_platform: bool = torch._C._valgrind_supported_platform()
        else:
            print('Callgrind bindings are not present in `torch._C`. JIT-ing bindings.')
            self._bindings_module = cpp_jit.get_compat_bindings()
            assert all((hasattr(self._bindings_module, symbol) for symbol in valgrind_symbols))
            self._supported_platform = self._bindings_module._valgrind_supported_platform()
        self._commands_available: Dict[str, bool] = {}
        if self._supported_platform:
            for cmd in ('valgrind', 'callgrind_control', 'callgrind_annotate'):
                self._commands_available[cmd] = not subprocess.run(['which', cmd], capture_output=True, check=False).returncode
        self._build_type: Optional[str] = None
        build_search = re.search('BUILD_TYPE=(.+),', torch.__config__.show())
        if build_search is not None:
            self._build_type = build_search.groups()[0].split(',')[0]

    def _validate(self) -> None:
        if not self._supported_platform:
            raise OSError('Valgrind is not supported on this platform.')
        missing_cmds = [cmd for cmd, available in self._commands_available.items() if not available]
        if missing_cmds:
            raise OSError('Missing: ' + ', '.join(missing_cmds))

    def collect_callgrind(self, task_spec: common.TaskSpec, globals: Dict[str, Any], *, number: int, repeats: int, collect_baseline: bool, is_python: bool, retain_out_file: bool) -> Tuple[CallgrindStats, ...]:
        """Collect stats, and attach a reference run which can be used to filter interpreter overhead."""
        self._validate()
        assert is_python or not collect_baseline
        *task_stats, baseline_stats = self._invoke(task_spec=task_spec, globals=globals, number=number, repeats=repeats, collect_baseline=collect_baseline, is_python=is_python, retain_out_file=retain_out_file)
        assert len(task_stats) == repeats
        return tuple((CallgrindStats(task_spec=task_spec, number_per_run=number, built_with_debug_symbols=self._build_type == 'RelWithDebInfo', baseline_inclusive_stats=baseline_stats[0], baseline_exclusive_stats=baseline_stats[1], stmt_inclusive_stats=stmt_inclusive_stats, stmt_exclusive_stats=stmt_exclusive_stats, stmt_callgrind_out=out_contents) for stmt_inclusive_stats, stmt_exclusive_stats, out_contents in task_stats))

    def _invoke(self, *, task_spec: common.TaskSpec, globals: Dict[str, Any], number: int, repeats: int, collect_baseline: bool, is_python: bool, retain_out_file: bool) -> Tuple[Tuple[FunctionCounts, FunctionCounts, Optional[str]], ...]:
        """Core invocation method for Callgrind collection.

        Valgrind operates by effectively replacing the CPU with an emulated
        version which allows it to instrument any code at the cost of severe
        performance degradation. This has the practical effect that in order
        to collect Callgrind statistics, a new process has to be created
        running under `valgrind`. The steps for this process are:

        1) Create a scratch directory.
        2) Codegen a run script. (_ValgrindWrapper._construct_script)
            Inside the run script:
                * Validate that Python and torch match the parent process
                * Validate that it is indeed running under valgrind
                * Execute `setup` and warm up `stmt`
                * Begin collecting stats
                * Run the `stmt` loop
                * Stop collecting stats
        3) Parse the run results.
        4) Cleanup the scratch directory.
        """
        working_dir = common._make_temp_dir(prefix='callgrind')
        data_dir = os.path.join(working_dir, 'data')
        script_file = os.path.join(working_dir, 'timer_callgrind.py')
        callgrind_out = os.path.join(working_dir, 'callgrind.out')
        error_log = os.path.join(working_dir, 'error.txt')
        stat_log = os.path.join(working_dir, 'callgrind_stat.txt')
        stdout_stderr_log = os.path.join(working_dir, 'stdout_stderr.log')

        def run(args: List[str], **kwargs: Any) -> Tuple[CompletedProcessType, str]:
            f_stdout_stderr = open(stdout_stderr_log, 'wb')
            try:
                invocation = subprocess.run(args, stdout=f_stdout_stderr, stderr=subprocess.STDOUT, **kwargs)
                with open(stdout_stderr_log) as f:
                    return (invocation, f.read())
            finally:
                f_stdout_stderr.close()
        try:
            if is_python:
                if self._bindings_module is not None:
                    shutil.copy(self._bindings_module.__file__, os.path.join(working_dir, os.path.split(self._bindings_module.__file__)[1]))
                script_file = os.path.join(working_dir, 'timer_callgrind.py')
                with open(script_file, 'w') as f:
                    f.write(self._construct_script(task_spec, globals=GlobalsBridge(globals, data_dir), number=number, repeats=repeats, collect_baseline=collect_baseline, error_log=error_log, stat_log=stat_log, bindings=self._bindings_module))
                run_loop_cmd = ['python', script_file]
            else:
                assert not collect_baseline
                run_loop_exec = cpp_jit.compile_callgrind_template(stmt=task_spec.stmt, setup=task_spec.setup, global_setup=task_spec.global_setup)
                run_loop_cmd = [run_loop_exec, '--number', str(number), '--number-warmup', str(min(number, 10)), '--repeats', str(repeats), '--number-threads', str(task_spec.num_threads)]
            valgrind_invocation, valgrind_invocation_output = run(['valgrind', '--tool=callgrind', f'--callgrind-out-file={callgrind_out}', '--dump-line=yes', '--dump-instr=yes', '--instr-atstart=yes', '--collect-atstart=no'] + run_loop_cmd)
            if valgrind_invocation.returncode:
                error_report = ''
                if os.path.exists(error_log):
                    with open(error_log) as f:
                        error_report = f.read()
                if not error_report:
                    error_report = 'Unknown error.\n' + valgrind_invocation_output
                raise OSError(f'Failed to collect callgrind profile:\n{error_report}')

            def parse_output(fpath: str, inclusive: bool) -> FunctionCounts:
                annotate_invocation, annotate_invocation_output = run(['callgrind_annotate', f'--inclusive={('yes' if inclusive else 'no')}', '--threshold=100', '--show-percs=no', fpath], check=True)
                total_pattern = re.compile('^([0-9,]+)\\s+PROGRAM TOTALS')
                begin_pattern = re.compile('Ir\\s+file:function')
                function_pattern = re.compile('^\\s*([0-9,]+)\\s+(.+:.+)$')

                class ScanState(enum.Enum):
                    SCANNING_FOR_TOTAL = 0
                    SCANNING_FOR_START = 1
                    PARSING = 2
                scan_state = ScanState.SCANNING_FOR_TOTAL
                fn_counts = []
                for l in annotate_invocation_output.splitlines(keepends=False):
                    if scan_state == ScanState.SCANNING_FOR_TOTAL:
                        total_match = total_pattern.match(l)
                        if total_match:
                            program_totals = int(total_match.groups()[0].replace(',', ''))
                            scan_state = ScanState.SCANNING_FOR_START
                    elif scan_state == ScanState.SCANNING_FOR_START:
                        if begin_pattern.match(l):
                            scan_state = ScanState.PARSING
                    else:
                        assert scan_state == ScanState.PARSING
                        fn_match = function_pattern.match(l)
                        if fn_match:
                            ir_str, file_function = fn_match.groups()
                            ir = int(ir_str.replace(',', ''))
                            if ir == program_totals:
                                continue
                            fn_counts.append(FunctionCount(ir, file_function))
                        elif re.match('-+', l):
                            continue
                        else:
                            break
                assert scan_state == ScanState.PARSING, f'Failed to parse {fpath}'
                return FunctionCounts(tuple(sorted(fn_counts, reverse=True)), inclusive=inclusive)

            def read_results(i: int) -> Tuple[FunctionCounts, FunctionCounts, Optional[str]]:
                if i == repeats and (not collect_baseline):
                    return (FunctionCounts((), inclusive=True), FunctionCounts((), inclusive=False), None)
                fpath = f'{callgrind_out}.{i + 1}'
                callgrind_out_contents: Optional[str] = None
                if retain_out_file:
                    with open(fpath) as f:
                        callgrind_out_contents = f.read()
                return (parse_output(fpath, inclusive=True), parse_output(fpath, inclusive=False), callgrind_out_contents)
            return tuple((read_results(i) for i in range(repeats + 1)))
        finally:
            shutil.rmtree(working_dir)

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