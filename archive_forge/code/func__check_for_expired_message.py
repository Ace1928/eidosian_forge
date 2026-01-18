import greenlet
import time
from curtsies import events
from ..translations import _
from ..repl import Interaction
from ..curtsiesfrontend.events import RefreshRequestEvent
from ..curtsiesfrontend.manual_readline import edit_keys
def _check_for_expired_message(self):
    if self._message and time.time() > self.message_start_time + self.message_time:
        self._message = ''