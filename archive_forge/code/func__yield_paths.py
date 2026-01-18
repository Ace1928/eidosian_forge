from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar, Union
from langchain_community.document_loaders.blob_loaders.schema import Blob, BlobLoader
def _yield_paths(self) -> Iterable[Path]:
    """Yield paths that match the requested pattern."""
    if self.path.is_file():
        yield self.path
        return
    paths = self.path.glob(self.glob)
    for path in paths:
        if self.exclude:
            if any((path.match(glob) for glob in self.exclude)):
                continue
        if path.is_file():
            if self.suffixes and path.suffix not in self.suffixes:
                continue
            yield path