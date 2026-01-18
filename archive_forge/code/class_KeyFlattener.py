from collections import defaultdict
from string import Formatter
from typing import Any, Dict, Optional
from ._interfaces import LogEvent
class KeyFlattener:
    """
    A L{KeyFlattener} computes keys for the things within curly braces in
    PEP-3101-style format strings as parsed by L{string.Formatter.parse}.
    """

    def __init__(self) -> None:
        """
        Initialize a L{KeyFlattener}.
        """
        self.keys: Dict[str, int] = defaultdict(lambda: 0)

    def flatKey(self, fieldName: str, formatSpec: Optional[str], conversion: Optional[str]) -> str:
        """
        Compute a string key for a given field/format/conversion.

        @param fieldName: A format field name.
        @param formatSpec: A format spec.
        @param conversion: A format field conversion type.

        @return: A key specific to the given field, format and conversion, as
            well as the occurrence of that combination within this
            L{KeyFlattener}'s lifetime.
        """
        if formatSpec is None:
            formatSpec = ''
        if conversion is None:
            conversion = ''
        result = '{fieldName}!{conversion}:{formatSpec}'.format(fieldName=fieldName, formatSpec=formatSpec, conversion=conversion)
        self.keys[result] += 1
        n = self.keys[result]
        if n != 1:
            result += '/' + str(self.keys[result])
        return result