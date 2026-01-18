from typing import Any
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.io import exists, join, makedirs, rm
from fugue.collections.partition import PartitionSpec
from fugue.collections.yielded import PhysicalYielded
from fugue.constants import FUGUE_CONF_WORKFLOW_CHECKPOINT_PATH
from fugue.dataframe import DataFrame
from fugue.exceptions import FugueWorkflowCompileError, FugueWorkflowRuntimeError
from fugue.execution.execution_engine import ExecutionEngine
class CheckpointPath(object):

    def __init__(self, engine: ExecutionEngine):
        self._engine = engine
        self._log = engine.log
        self._path = engine.conf.get(FUGUE_CONF_WORKFLOW_CHECKPOINT_PATH, '').strip()
        self._temp_path = ''

    @property
    def execution_engine(self) -> ExecutionEngine:
        return self._engine

    def init_temp_path(self, execution_id: str) -> str:
        if self._path == '':
            self._temp_path = ''
            return ''
        self._temp_path = join(self._path, execution_id)
        makedirs(self._temp_path, exist_ok=True)
        return self._temp_path

    def remove_temp_path(self):
        if self._temp_path != '':
            try:
                rm(self._temp_path, recursive=True)
            except Exception as e:
                self._log.info('Unable to remove ' + self._temp_path, e)

    def get_temp_file(self, obj_id: str, permanent: bool) -> str:
        path = self._path if permanent else self._temp_path
        assert_or_throw(path != '', FugueWorkflowRuntimeError(f'{FUGUE_CONF_WORKFLOW_CHECKPOINT_PATH} is not set'))
        return join(path, obj_id + '.parquet')

    def get_table_name(self, obj_id: str, permanent: bool) -> str:
        path = self._path if permanent else self._temp_path
        return 'temp_' + to_uuid(path, obj_id)[:5]

    def temp_file_exists(self, path: str) -> bool:
        try:
            return exists(path)
        except Exception:
            return False