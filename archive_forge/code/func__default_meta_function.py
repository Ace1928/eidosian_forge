import itertools
import re
from typing import Any, Callable, Generator, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_community.document_loaders.web_base import WebBaseLoader
def _default_meta_function(meta: dict, _content: Any) -> dict:
    return {'source': meta['loc'], **meta}