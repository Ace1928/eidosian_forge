import os
import signal
import subprocess
import time
from wandb_watchdog.utils import echo, has_attribute
from wandb_watchdog.events import PatternMatchingEventHandler
class LoggerTrick(Trick):
    """A simple trick that does only logs events."""

    def on_any_event(self, event):
        pass

    @echo.echo
    def on_modified(self, event):
        pass

    @echo.echo
    def on_deleted(self, event):
        pass

    @echo.echo
    def on_created(self, event):
        pass

    @echo.echo
    def on_moved(self, event):
        pass