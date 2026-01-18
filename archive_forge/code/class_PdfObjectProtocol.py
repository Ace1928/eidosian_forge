from abc import abstractmethod
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Tuple, Union
from ._utils import StrByteType, StreamType
class PdfObjectProtocol(Protocol):
    indirect_reference: Any

    def clone(self, pdf_dest: Any, force_duplicate: bool=False, ignore_fields: Union[Tuple[str, ...], List[str], None]=()) -> Any:
        ...

    def _reference_clone(self, clone: Any, pdf_dest: Any) -> Any:
        ...

    def get_object(self) -> Optional['PdfObjectProtocol']:
        ...

    def hash_value(self) -> bytes:
        ...

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        ...