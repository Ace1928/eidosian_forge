from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar, Union
from langchain_community.document_loaders.blob_loaders.schema import Blob, BlobLoader
def _with_tqdm(iterable: Iterable[T]) -> Iterator[T]:
    """Wrap an iterable in a tqdm progress bar."""
    return tqdm(iterable, total=length_func())