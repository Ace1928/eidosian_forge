import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Generator, Iterable, List, TextIO, Tuple
import pytest
from spacy.lang.en import English
from spacy.training import Example, PlainTextCorpus
from spacy.util import make_tempdir
def _examples_to_tokens(examples: Iterable[Example]) -> Tuple[List[List[str]], List[List[str]]]:
    reference = []
    predicted = []
    for eg in examples:
        reference.append([t.text for t in eg.reference])
        predicted.append([t.text for t in eg.predicted])
    return (reference, predicted)