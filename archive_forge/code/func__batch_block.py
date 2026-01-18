import itertools
import re
from typing import Any, Callable, Generator, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_community.document_loaders.web_base import WebBaseLoader
def _batch_block(iterable: Iterable, size: int) -> Generator[List[dict], None, None]:
    it = iter(iterable)
    while (item := list(itertools.islice(it, size))):
        yield item