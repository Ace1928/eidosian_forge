from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Tuple, Union
from watchdog.observers.inotify_c import Inotify, InotifyEvent
from watchdog.utils import BaseThread
from watchdog.utils.delayed_queue import DelayedQueue
def _group_events(self, event_list):
    """Group any matching move events"""
    grouped: List[Union[InotifyEvent, Tuple[InotifyEvent, InotifyEvent]]] = []
    for inotify_event in event_list:
        logger.debug('in-event %s', inotify_event)

        def matching_from_event(event):
            return not isinstance(event, tuple) and event.is_moved_from and (event.cookie == inotify_event.cookie)
        if inotify_event.is_moved_to:
            for index, event in enumerate(grouped):
                if matching_from_event(event):
                    if TYPE_CHECKING:
                        assert not isinstance(event, tuple)
                    grouped[index] = (event, inotify_event)
                    break
            else:
                from_event = self._queue.remove(matching_from_event)
                if from_event is not None:
                    grouped.append((from_event, inotify_event))
                else:
                    logger.debug('could not find matching move_from event')
                    grouped.append(inotify_event)
        else:
            grouped.append(inotify_event)
    return grouped