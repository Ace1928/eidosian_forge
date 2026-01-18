from __future__ import annotations
import warnings
from typing import cast
from typing import Generator
from typing import Mapping
from typing import NoReturn
from typing import Sequence
from typing import Tuple
from typing import Union
from ._hooks import HookImpl
from ._result import HookCallError
from ._result import Result
from ._warnings import PluggyTeardownRaisedWarning
def _raise_wrapfail(wrap_controller: Generator[None, Result[object], None] | Generator[None, object, object], msg: str) -> NoReturn:
    co = wrap_controller.gi_code
    raise RuntimeError('wrap_controller at %r %s:%d %s' % (co.co_name, co.co_filename, co.co_firstlineno, msg))