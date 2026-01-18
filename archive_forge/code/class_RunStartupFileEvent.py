import time
from typing import Sequence
import curtsies.events
class RunStartupFileEvent(curtsies.events.Event):
    """Request to run the startup file."""