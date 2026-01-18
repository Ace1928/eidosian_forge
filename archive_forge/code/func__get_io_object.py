from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union
from ._events import (
from ._headers import get_comma_header, has_expect_100_continue, set_comma_header
from ._readers import READERS, ReadersType
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import (  # Import the internal things we need
from ._writers import WRITERS, WritersType
def _get_io_object(self, role: Type[Sentinel], event: Optional[Event], io_dict: Union[ReadersType, WritersType]) -> Optional[Callable[..., Any]]:
    state = self._cstate.states[role]
    if state is SEND_BODY:
        framing_type, args = _body_framing(cast(bytes, self._request_method), cast(Union[Request, Response], event))
        return io_dict[SEND_BODY][framing_type](*args)
    else:
        return io_dict.get((role, state))