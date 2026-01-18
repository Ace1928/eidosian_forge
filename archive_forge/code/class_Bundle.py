from typing import Dict, List, Optional, Sequence, Tuple, Union
from .pack import PackData, write_pack_data
class Bundle:
    version: Optional[int]
    capabilities: Dict[str, str]
    prerequisites: List[Tuple[bytes, str]]
    references: Dict[str, bytes]
    pack_data: Union[PackData, Sequence[bytes]]

    def __repr__(self) -> str:
        return f'<{type(self).__name__}(version={self.version}, capabilities={self.capabilities}, prerequisites={self.prerequisites}, references={self.references})>'

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.version != other.version:
            return False
        if self.capabilities != other.capabilities:
            return False
        if self.prerequisites != other.prerequisites:
            return False
        if self.references != other.references:
            return False
        if self.pack_data != other.pack_data:
            return False
        return True