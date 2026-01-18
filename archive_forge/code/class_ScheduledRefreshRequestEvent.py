import time
from typing import Sequence
import curtsies.events
class ScheduledRefreshRequestEvent(curtsies.events.ScheduledEvent):
    """Request to refresh the REPL display at some point in the future

    Used to schedule the disappearance of status bar message that only shows
    for a few seconds"""

    def __init__(self, when: float) -> None:
        super().__init__(when)

    def __repr__(self) -> str:
        return '<RefreshRequestEvent for {} seconds from now>'.format(self.when - time.time())