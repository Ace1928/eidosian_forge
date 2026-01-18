import sys
import threading
from IPython import get_ipython
from IPython.core.ultratb import AutoFormattedTB
from logging import error, debug
Create a new job from a callable object.

        Any positional arguments and keyword args given to this constructor
        after the initial callable are passed directly to it.