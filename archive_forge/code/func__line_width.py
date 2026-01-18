from collections import deque, OrderedDict
from typing import Union, Optional, Set, Any, Dict, List, Tuple
from datetime import timedelta
import functools
import math
import time
import re
import shutil
import json
from parlai.core.message import Message
from parlai.utils.strings import colorize
import parlai.utils.logging as logging
def _line_width():
    try:
        __IPYTHON__
        return 128
    except NameError:
        return shutil.get_terminal_size((88, 24)).columns