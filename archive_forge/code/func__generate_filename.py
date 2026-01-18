from typing import Any, Dict, Optional
from ray.data.block import Block
from ray.util.annotations import PublicAPI
def _generate_filename(self, file_id: str) -> str:
    filename = ''
    if self._dataset_uuid is not None:
        filename += f'{self._dataset_uuid}_'
    filename += file_id
    if self._file_format is not None:
        filename += f'.{self._file_format}'
    return filename