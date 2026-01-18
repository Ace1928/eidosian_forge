import abc
import dataclasses
import os
from typing import Any, Dict, List, TYPE_CHECKING
import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
class _FilesystemSaver(_WorkflowSaver):
    """Save data to the filesystem.

    The ExecutableGroupResult's constituent parts will be saved to disk as they become
    available. Within the "{base_data_dir}/{run_id}" directory we save:
        - The `cg.QuantumRuntimeConfiguration` at the start of the execution as a record
          of *how* the executable group was run.
        - A `cg.SharedRuntimeInfo` which is updated throughout the run.
        - An `cg.ExecutableResult` for each `cg.QuantumExecutable` as they become available.
        - A `cg.ExecutableGroupResultFilesystemRecord` which is updated throughout the run.

    Args:
        base_data_dir: Each data file will be written to the "{base_data_dir}/{run_id}/" directory,
            which must not already exist.
        run_id: Each data file will be written to the "{base_data_dir}/{run_id}/" directory,
            which must not already exist.
    """

    def __init__(self, base_data_dir, run_id):
        self.base_data_dir = base_data_dir
        self.run_id = run_id
        self._data_dir = f'{self.base_data_dir}/{self.run_id}'
        self._egr_record = None

    @property
    def data_dir(self) -> str:
        """The data directory, namely '{base_data_dir}/{run_id}"""
        return self._data_dir

    @property
    def egr_record(self) -> ExecutableGroupResultFilesystemRecord:
        """The `cg.ExecutablegroupResultFilesystemRecord` keeping track of all the paths for saved
        files."""
        return self._egr_record

    def initialize(self, rt_config: 'cg.QuantumRuntimeConfiguration', shared_rt_info: 'cg.SharedRuntimeInfo'):
        """Initialize the filesystem for data saving

        Args:
            rt_config: The immutable `cg.QuantumRuntimeConfiguation` for this run. This is written
                once during this initialization.
            shared_rt_info: The initial `cg.SharedRuntimeInfo` to be saved to a file.
        """
        os.makedirs(self._data_dir, exist_ok=False)
        self._egr_record = ExecutableGroupResultFilesystemRecord(runtime_configuration_path='QuantumRuntimeConfiguration.json.gz', shared_runtime_info_path='SharedRuntimeInfo.json.gz', executable_result_paths=[], run_id=self.run_id)
        cirq.to_json_gzip(rt_config, f'{self._data_dir}/{self._egr_record.runtime_configuration_path}')
        _update_updatable_files(self._egr_record, shared_rt_info, self._data_dir)

    def consume_result(self, exe_result: 'cg.ExecutableResult', shared_rt_info: 'cg.SharedRuntimeInfo'):
        """Save an `cg.ExecutableResult` that has been completed.

        Args:
            exe_result: The completed `cg.ExecutableResult` to be saved to
                'ExecutableResult.{i}.json.gz'
            shared_rt_info: The current `cg.SharedRuntimeInfo` to update SharedRuntimeInfo.json.gz.
        """
        i = exe_result.runtime_info.execution_index
        exe_result_path = f'ExecutableResult.{i}.json.gz'
        cirq.to_json_gzip(exe_result, f'{self._data_dir}/{exe_result_path}')
        self._egr_record.executable_result_paths.append(exe_result_path)
        _update_updatable_files(self._egr_record, shared_rt_info, self._data_dir)

    def finalize(self, shared_rt_info: 'cg.SharedRuntimeInfo'):
        """Called at the end of a workflow execution to finalize data saving.

        Args:
            shared_rt_info: The final `cg.SharedRuntimeInfo` to be saved or updated.
        """
        _update_updatable_files(self.egr_record, shared_rt_info, self._data_dir)