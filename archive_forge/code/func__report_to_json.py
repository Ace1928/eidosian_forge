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
def _report_to_json(report: BaseReport) -> Dict[str, Any]:
    """Return the contents of this report as a dict of builtin entries,
    suitable for serialization.

    This was originally the serialize_report() function from xdist (ca03269).
    """

    def serialize_repr_entry(entry: Union[ReprEntry, ReprEntryNative]) -> Dict[str, Any]:
        data = dataclasses.asdict(entry)
        for key, value in data.items():
            if hasattr(value, '__dict__'):
                data[key] = dataclasses.asdict(value)
        entry_data = {'type': type(entry).__name__, 'data': data}
        return entry_data

    def serialize_repr_traceback(reprtraceback: ReprTraceback) -> Dict[str, Any]:
        result = dataclasses.asdict(reprtraceback)
        result['reprentries'] = [serialize_repr_entry(x) for x in reprtraceback.reprentries]
        return result

    def serialize_repr_crash(reprcrash: Optional[ReprFileLocation]) -> Optional[Dict[str, Any]]:
        if reprcrash is not None:
            return dataclasses.asdict(reprcrash)
        else:
            return None

    def serialize_exception_longrepr(rep: BaseReport) -> Dict[str, Any]:
        assert rep.longrepr is not None
        longrepr = cast(ExceptionRepr, rep.longrepr)
        result: Dict[str, Any] = {'reprcrash': serialize_repr_crash(longrepr.reprcrash), 'reprtraceback': serialize_repr_traceback(longrepr.reprtraceback), 'sections': longrepr.sections}
        if isinstance(longrepr, ExceptionChainRepr):
            result['chain'] = []
            for repr_traceback, repr_crash, description in longrepr.chain:
                result['chain'].append((serialize_repr_traceback(repr_traceback), serialize_repr_crash(repr_crash), description))
        else:
            result['chain'] = None
        return result
    d = report.__dict__.copy()
    if hasattr(report.longrepr, 'toterminal'):
        if hasattr(report.longrepr, 'reprtraceback') and hasattr(report.longrepr, 'reprcrash'):
            d['longrepr'] = serialize_exception_longrepr(report)
        else:
            d['longrepr'] = str(report.longrepr)
    else:
        d['longrepr'] = report.longrepr
    for name in d:
        if isinstance(d[name], os.PathLike):
            d[name] = os.fspath(d[name])
        elif name == 'result':
            d[name] = None
    return d