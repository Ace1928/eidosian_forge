from abc import abstractmethod
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Tuple, Union
from ._utils import StrByteType, StreamType
class PdfReaderProtocol(PdfCommonDocProtocol, Protocol):

    @property
    @abstractmethod
    def xref(self) -> Dict[int, Dict[int, Any]]:
        ...

    @property
    @abstractmethod
    def trailer(self) -> Dict[str, Any]:
        ...