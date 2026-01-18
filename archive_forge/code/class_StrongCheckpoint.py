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
class StrongCheckpoint(Checkpoint):

    def __init__(self, storage_type: str, obj_id: str, deterministic: bool, permanent: bool, lazy: bool=False, partition: Any=None, single: bool=False, namespace: Any=None, **save_kwargs: Any):
        super().__init__(deterministic=deterministic, permanent=permanent, lazy=lazy, fmt='', partition=PartitionSpec(partition), single=single, namespace=namespace, save_kwargs=dict(save_kwargs))
        self._yield_func: Any = None
        self._obj_id = to_uuid(obj_id, namespace)
        self._yielded = PhysicalYielded(self._obj_id, storage_type=storage_type)

    def run(self, df: DataFrame, path: 'CheckpointPath') -> DataFrame:
        if self._yielded.storage_type == 'file':
            fpath = path.get_temp_file(self._obj_id, self.permanent)
            if not self.deterministic or not path.temp_file_exists(fpath):
                path.execution_engine.save_df(df=df, path=fpath, format_hint=self.kwargs['fmt'], mode='overwrite', partition_spec=self.kwargs['partition'], force_single=self.kwargs['single'], **self.kwargs['save_kwargs'])
            result = path.execution_engine.load_df(path=fpath, format_hint=self.kwargs['fmt'])
            self._yielded.set_value(fpath)
        else:
            tb = path.get_table_name(self._obj_id, self.permanent)
            if not self.deterministic or not path.execution_engine.sql_engine.table_exists(tb):
                path.execution_engine.sql_engine.save_table(df=df, table=tb, partition_spec=self.kwargs['partition'], **self.kwargs['save_kwargs'])
            result = path.execution_engine.sql_engine.load_table(tb)
            self._yielded.set_value(tb)
        return result

    @property
    def yielded(self) -> PhysicalYielded:
        assert_or_throw(self.permanent, lambda: FugueWorkflowCompileError(f'yield is not allowed for {self}'))
        return self._yielded

    @property
    def is_null(self) -> bool:
        return False