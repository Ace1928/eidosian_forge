import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
class SystemExitFromCodeRunner(SystemExit):
    """If this class is returned, a SystemExit happened while in the code
    greenlet"""