import posixpath
from typing import TYPE_CHECKING, Optional
from ray.util.annotations import DeveloperAPI
def _get_write_path_for_block(self, base_path: str, *, filesystem: Optional['pyarrow.fs.FileSystem']=None, dataset_uuid: Optional[str]=None, task_index: Optional[int]=None, block_index: Optional[int]=None, file_format: Optional[str]=None) -> str:
    assert task_index is not None
    if block_index is not None:
        suffix = f'{dataset_uuid}_{task_index:06}_{block_index:06}.{file_format}'
    else:
        suffix = f'{dataset_uuid}_{task_index:06}.{file_format}'
    return posixpath.join(base_path, suffix)