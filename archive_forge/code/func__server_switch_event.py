from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union
from ._events import (
from ._headers import get_comma_header, has_expect_100_continue, set_comma_header
from ._readers import READERS, ReadersType
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import (  # Import the internal things we need
from ._writers import WRITERS, WritersType
def _server_switch_event(self, event: Event) -> Optional[Type[Sentinel]]:
    if type(event) is InformationalResponse and event.status_code == 101:
        return _SWITCH_UPGRADE
    if type(event) is Response:
        if _SWITCH_CONNECT in self._cstate.pending_switch_proposals and 200 <= event.status_code < 300:
            return _SWITCH_CONNECT
    return None