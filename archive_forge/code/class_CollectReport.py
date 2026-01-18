import dataclasses
from io import StringIO
import os
from pprint import pprint
from typing import Any
from typing import cast
from typing import Dict
from typing import final
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from _pytest._code.code import ExceptionChainRepr
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ExceptionRepr
from _pytest._code.code import ReprEntry
from _pytest._code.code import ReprEntryNative
from _pytest._code.code import ReprExceptionInfo
from _pytest._code.code import ReprFileLocation
from _pytest._code.code import ReprFuncArgs
from _pytest._code.code import ReprLocals
from _pytest._code.code import ReprTraceback
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.config import Config
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import skip
@final
class CollectReport(BaseReport):
    """Collection report object.

    Reports can contain arbitrary extra attributes.
    """
    when = 'collect'

    def __init__(self, nodeid: str, outcome: "Literal['passed', 'failed', 'skipped']", longrepr: Union[None, ExceptionInfo[BaseException], Tuple[str, int, str], str, TerminalRepr], result: Optional[List[Union[Item, Collector]]], sections: Iterable[Tuple[str, str]]=(), **extra) -> None:
        self.nodeid = nodeid
        self.outcome = outcome
        self.longrepr = longrepr
        self.result = result or []
        self.sections = list(sections)
        self.__dict__.update(extra)

    @property
    def location(self) -> Optional[Tuple[str, Optional[int], str]]:
        return (self.fspath, None, self.fspath)

    def __repr__(self) -> str:
        return f'<CollectReport {self.nodeid!r} lenresult={len(self.result)} outcome={self.outcome!r}>'