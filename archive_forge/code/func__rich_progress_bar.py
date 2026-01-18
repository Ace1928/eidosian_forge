import functools
from typing import Callable, Generator, Iterable, Iterator, Optional, Tuple
from pip._vendor.rich.progress import (
from pip._internal.utils.logging import get_indentation
def _rich_progress_bar(iterable: Iterable[bytes], *, bar_type: str, size: int) -> Generator[bytes, None, None]:
    assert bar_type == 'on', 'This should only be used in the default mode.'
    if not size:
        total = float('inf')
        columns: Tuple[ProgressColumn, ...] = (TextColumn('[progress.description]{task.description}'), SpinnerColumn('line', speed=1.5), FileSizeColumn(), TransferSpeedColumn(), TimeElapsedColumn())
    else:
        total = size
        columns = (TextColumn('[progress.description]{task.description}'), BarColumn(), DownloadColumn(), TransferSpeedColumn(), TextColumn('eta'), TimeRemainingColumn())
    progress = Progress(*columns, refresh_per_second=30)
    task_id = progress.add_task(' ' * (get_indentation() + 2), total=total)
    with progress:
        for chunk in iterable:
            yield chunk
            progress.update(task_id, advance=len(chunk))