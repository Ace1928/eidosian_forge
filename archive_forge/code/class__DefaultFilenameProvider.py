from typing import Any, Dict, Optional
from ray.data.block import Block
from ray.util.annotations import PublicAPI
class _DefaultFilenameProvider(FilenameProvider):

    def __init__(self, dataset_uuid: Optional[str]=None, file_format: Optional[str]=None):
        self._dataset_uuid = dataset_uuid
        self._file_format = file_format

    def get_filename_for_block(self, block: Block, task_index: int, block_index: int) -> str:
        file_id = f'{task_index:06}_{block_index:06}'
        return self._generate_filename(file_id)

    def get_filename_for_row(self, row: Dict[str, Any], task_index: int, block_index: int, row_index: int) -> str:
        file_id = f'{task_index:06}_{block_index:06}_{row_index:06}'
        return self._generate_filename(file_id)

    def _generate_filename(self, file_id: str) -> str:
        filename = ''
        if self._dataset_uuid is not None:
            filename += f'{self._dataset_uuid}_'
        filename += file_id
        if self._file_format is not None:
            filename += f'.{self._file_format}'
        return filename