import warnings
from typing import Dict, Iterable, Iterator, List, Tuple, Union, cast
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
def _consume_os(tags: List[str]) -> Iterator[str]:
    while tags and tags[0] == 'O':
        yield tags.pop(0)