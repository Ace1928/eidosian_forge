from typing import cast, Dict, Optional, Set, Tuple, Type, Union
from ._events import *
from ._util import LocalProtocolError, Sentinel
class MIGHT_SWITCH_PROTOCOL(Sentinel, metaclass=Sentinel):
    pass