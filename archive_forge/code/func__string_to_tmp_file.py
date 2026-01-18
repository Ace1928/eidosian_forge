import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Generator, Iterable, List, TextIO, Tuple
import pytest
from spacy.lang.en import English
from spacy.training import Example, PlainTextCorpus
from spacy.util import make_tempdir
@contextmanager
def _string_to_tmp_file(s: str) -> Generator[Path, None, None]:
    with make_tempdir() as d:
        file_path = Path(d) / 'string.txt'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(s)
        yield file_path