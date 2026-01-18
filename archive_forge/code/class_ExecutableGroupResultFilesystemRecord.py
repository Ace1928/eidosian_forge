import abc
import dataclasses
import os
from typing import Any, Dict, List, TYPE_CHECKING
import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
@dataclasses.dataclass
class ExecutableGroupResultFilesystemRecord:
    """Filename references to the constituent parts of a `cg.ExecutableGroupResult`.

    Args:
        runtime_configuration_path: A filename pointing to the `runtime_configuration` value.
        shared_runtime_info_path: A filename pointing to the `shared_runtime_info` value.
        executable_result_paths: A list of filenames pointing to the `executable_results` values.
        run_id: The unique `str` identifier from this run. This is used to locate the other
            values on disk.
    """
    runtime_configuration_path: str
    shared_runtime_info_path: str
    executable_result_paths: List[str]
    run_id: str

    @classmethod
    def from_json(cls, *, run_id: str, base_data_dir: str='.') -> 'ExecutableGroupResultFilesystemRecord':
        fn = f'{base_data_dir}/{run_id}/ExecutableGroupResultFilesystemRecord.json.gz'
        egr_record = cirq.read_json_gzip(fn)
        if not isinstance(egr_record, cls):
            raise ValueError(f'The file located at {fn} is not an `ExecutableGroupFilesystemRecord`.')
        if egr_record.run_id != run_id:
            raise ValueError(f'The loaded run_id {run_id} does not match the provided run_id {run_id}')
        return egr_record

    def load(self, *, base_data_dir: str='.') -> 'cg.ExecutableGroupResult':
        """Using the filename references in this dataclass, load a `cg.ExecutableGroupResult`
        from its constituent parts.

        Args:
            base_data_dir: The base data directory. Files should be found at
                {base_data_dir}/{run_id}/{this class's paths}
        """
        data_dir = f'{base_data_dir}/{self.run_id}'
        from cirq_google.workflow.quantum_runtime import ExecutableGroupResult
        return ExecutableGroupResult(runtime_configuration=cirq.read_json_gzip(f'{data_dir}/{self.runtime_configuration_path}'), shared_runtime_info=cirq.read_json_gzip(f'{data_dir}/{self.shared_runtime_info_path}'), executable_results=[cirq.read_json_gzip(f'{data_dir}/{exe_path}') for exe_path in self.executable_result_paths])

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')