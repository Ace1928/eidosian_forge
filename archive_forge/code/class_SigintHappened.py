import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
class SigintHappened:
    """If this class is returned, a SIGINT happened while the main greenlet"""