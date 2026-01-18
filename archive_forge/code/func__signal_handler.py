import signal
import sys
import traceback
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.wrappers import framework
def _signal_handler(unused_signal, unused_frame):
    while True:
        response = input('\nSIGINT received. Quit program? (Y/n): ').strip()
        if response in ('', 'Y', 'y'):
            sys.exit(0)
        elif response in ('N', 'n'):
            break