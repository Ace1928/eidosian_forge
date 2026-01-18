import time
import dateparser
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Callable
class LazyTimer:
    timers = {}

    def stop_timer(self, name):
        assert name in LazyTimer.timers
        LazyTimer.timers[name].stop()
        return LazyTimer.timers[name]

    def __getitem__(self, name, t=None, *args, **kwargs):
        return LazyTimer.timers.get(name, None)

    def __call__(self, name, t=None, *args, **kwargs):
        if name not in LazyTimer.timers:
            LazyTimer.timers[name] = LazyTime(*args, t=t, **kwargs)
        return LazyTimer.timers[name]