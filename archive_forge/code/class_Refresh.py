import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
class Refresh(RequestFromCodeRunner):
    """Running code would like the main loop to refresh the display"""