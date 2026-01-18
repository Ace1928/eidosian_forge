import os
import time
from threading import Thread, Lock
import sentry_sdk
from sentry_sdk.utils import logger
from sentry_sdk._types import TYPE_CHECKING
def _thread():
    while self._running:
        time.sleep(self.interval)
        if self._running:
            self.run()