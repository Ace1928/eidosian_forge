from typing import cast, Dict, Optional, Set, Tuple, Type, Union
from ._events import *
from ._util import LocalProtocolError, Sentinel
def _fire_event_triggered_transitions(self, role: Type[Sentinel], event_type: Union[Type[Event], Tuple[Type[Event], Type[Sentinel]]]) -> None:
    state = self.states[role]
    try:
        new_state = EVENT_TRIGGERED_TRANSITIONS[role][state][event_type]
    except KeyError:
        event_type = cast(Type[Event], event_type)
        raise LocalProtocolError("can't handle event type {} when role={} and state={}".format(event_type.__name__, role, self.states[role])) from None
    self.states[role] = new_state