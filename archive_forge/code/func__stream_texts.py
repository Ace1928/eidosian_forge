from itertools import chain
from pathlib import Path
from typing import Iterable, List, Optional, Union, cast
import srsly
import tqdm
from wasabi import msg
from ..tokens import Doc, DocBin
from ..util import ensure_path, load_model
from ..vocab import Vocab
from ._util import Arg, Opt, app, import_code, setup_gpu, walk_directory
def _stream_texts(paths: Iterable[Path]) -> Iterable[str]:
    """
    Yields strings from text files in paths.
    """
    for path in paths:
        with open(path, 'r') as fin:
            text = fin.read()
            yield text