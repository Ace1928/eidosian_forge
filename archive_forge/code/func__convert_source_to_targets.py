from __future__ import annotations
import abc
import tempfile
from pathlib import Path, PurePath
from lazyops.utils.pooler import ThreadPoolV2
from lazyops.utils.logs import default_logger, null_logger, Logger
from typing import Any, Union, Optional, Type, Iterable, Callable, Dict, List, Tuple, TypeVar, TYPE_CHECKING
def _convert_source_to_targets(self, source: InputSourceType, targets: List[str], source_filename: Optional[str]=None, target_output: Optional[Any]=None, **kwargs) -> List[OutputType]:
    """
        Convert the source to the targets

        source: /path/to/file.pdf
        targets: ['.docx', '.txt']
        """
    results = {}
    data_iterator = list(enumerate(targets))
    for item in self.pool.sync_iterate(self._convert_source_to_targets_one, data_iterator, source=source, source_filename=source_filename, target_output=target_output, return_ordered=False, **kwargs):
        index, output = item
        results[index] = output
    return [results[i] for i in range(len(results))]