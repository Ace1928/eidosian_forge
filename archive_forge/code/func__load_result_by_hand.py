import datetime
import glob
import re
import time
import uuid
from typing import List, cast, Any
import numpy as np
import pytest
import cirq
import cirq_google as cg
from cirq_google.workflow.quantum_executable_test import _get_quantum_executables, _get_example_spec
from cirq_google.workflow.quantum_runtime import _time_into_runtime_info
def _load_result_by_hand(tmpdir: str, run_id: str) -> cg.ExecutableGroupResult:
    """Load `ExecutableGroupResult` "by hand" without using
    `ExecutableGroupResultFilesystemRecord`."""
    rt_config = cirq.read_json_gzip(f'{tmpdir}/{run_id}/QuantumRuntimeConfiguration.json.gz')
    shared_rt_info = cirq.read_json_gzip(f'{tmpdir}/{run_id}/SharedRuntimeInfo.json.gz')
    fns = glob.glob(f'{tmpdir}/{run_id}/ExecutableResult.*.json.gz')
    fns = sorted(fns, key=lambda s: int(cast(Any, re.search('ExecutableResult\\.(\\d+)\\.json\\.gz$', s)).group(1)))
    assert len(fns) == 3
    exe_results: List[cg.ExecutableResult] = [cirq.read_json_gzip(fn) for fn in fns]
    return cg.ExecutableGroupResult(runtime_configuration=rt_config, shared_runtime_info=shared_rt_info, executable_results=exe_results)