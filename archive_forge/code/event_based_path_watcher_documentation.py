from __future__ import annotations
import os
import threading
from typing import Callable, Final, cast
from blinker import ANY, Signal
from watchdog import events
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch
from streamlit.logger import get_logger
from streamlit.util import repr_
from streamlit.watcher import util
Close this _MultiPathWatcher object forever.