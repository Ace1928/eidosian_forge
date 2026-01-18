from abc import abstractmethod
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Tuple, Union
from ._utils import StrByteType, StreamType
class PdfCommonDocProtocol(Protocol):

    @property
    def pdf_header(self) -> str:
        ...

    @property
    def pages(self) -> List[Any]:
        ...

    @property
    def root_object(self) -> PdfObjectProtocol:
        ...

    def get_object(self, indirect_reference: Any) -> Optional[PdfObjectProtocol]:
        ...

    @property
    def strict(self) -> bool:
        ...