from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Generator, Optional, Tuple
from rdflib.graph import Graph
from rdflib.plugins.serializers.nt import _nt_row
@contextmanager
def _start_new_file(file_no: int) -> Generator[Tuple[Path, BinaryIO], None, None]:
    if TYPE_CHECKING:
        assert output_dir is not None
    fp = Path(output_dir) / f'{file_name_stem}_{str(file_no).zfill(6)}.nt'
    with open(fp, 'wb') as fh:
        yield (fp, fh)