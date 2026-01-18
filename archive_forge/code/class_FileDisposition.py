from __future__ import annotations
from typing import TYPE_CHECKING
from coverage.types import TFileDisposition
class FileDisposition:
    """A simple value type for recording what to do with a file."""
    original_filename: str
    canonical_filename: str
    source_filename: str | None
    trace: bool
    reason: str
    file_tracer: FileTracer | None
    has_dynamic_filename: bool

    def __repr__(self) -> str:
        return f'<FileDisposition {self.canonical_filename!r}: trace={self.trace}>'