import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
class Wait(RequestFromCodeRunner):
    """Running code would like the main loop to run for a bit"""