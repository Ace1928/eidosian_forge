from 1.0.5, you can use a timeout of -1::
from sys import platform
from os import environ
from functools import wraps, partial
from kivy.context import register_context
from kivy.config import Config
from kivy.logger import Logger
from kivy.compat import clock as _default_time
import time
from threading import Event as ThreadingEvent
class ClockBaseFreeInterruptOnly(ClockBaseInterruptFreeBehavior, CyClockBaseFree):
    """The ``free_only`` kivy clock. See module for details.
    """

    def idle(self):
        fps = self._max_fps
        current = self.time()
        event = self._event
        if fps > 0:
            min_sleep = self.get_resolution()
            usleep = self.usleep
            undershoot = 4 / 5.0 * min_sleep
            min_t = self.get_min_free_timeout
            interupt_next_only = self.interupt_next_only
            sleeptime = 1 / fps - (current - self._last_tick)
            while sleeptime - undershoot > min_sleep:
                if event.is_set():
                    do_free = True
                else:
                    t = min_t()
                    if not t:
                        do_free = True
                    elif interupt_next_only:
                        do_free = False
                    else:
                        sleeptime = min(sleeptime, t - current)
                        do_free = sleeptime - undershoot <= min_sleep
                if do_free:
                    event.clear()
                    self._process_free_events(current)
                else:
                    event.wait(sleeptime - undershoot)
                current = self.time()
                sleeptime = 1 / fps - (current - self._last_tick)
        self._dt = current - self._last_tick
        self._last_tick = current
        event.clear()
        return current

    async def async_idle(self):
        fps = self._max_fps
        current = self.time()
        event = self._async_event
        if fps > 0:
            min_sleep = self.get_resolution()
            usleep = self.usleep
            undershoot = 4 / 5.0 * min_sleep
            min_t = self.get_min_free_timeout
            interupt_next_only = self.interupt_next_only
            sleeptime = 1 / fps - (current - self._last_tick)
            slept = False
            while sleeptime - undershoot > min_sleep:
                if event.is_set():
                    do_free = True
                else:
                    t = min_t()
                    if not t:
                        do_free = True
                    elif interupt_next_only:
                        do_free = False
                    else:
                        sleeptime = min(sleeptime, t - current)
                        do_free = sleeptime - undershoot <= min_sleep
                if do_free:
                    event.clear()
                    self._process_free_events(current)
                else:
                    slept = True
                    await self._async_wait_for(event.wait(), sleeptime - undershoot)
                current = self.time()
                sleeptime = 1 / fps - (current - self._last_tick)
            if not slept:
                await self._async_lib.sleep(0)
        else:
            await self._async_lib.sleep(0)
        self._dt = current - self._last_tick
        self._last_tick = current
        event.clear()
        return current