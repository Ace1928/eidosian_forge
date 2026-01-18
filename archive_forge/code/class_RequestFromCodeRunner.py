import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
class RequestFromCodeRunner:
    """Message from the code runner"""