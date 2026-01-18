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
def _ellipse(lst: List[str], max_display: int=5, sep: str='|') -> str:
    """
    Like join, but possibly inserts an ellipsis.

    :param lst: The list to join on
    :param int max_display: the number of items to display for ellipsing.
        If -1, shows all items
    :param string sep: the delimiter to join on
    """
    choices = list(lst)
    if max_display > 0 and len(choices) > max_display:
        ellipsis = '... ({} of {} shown)'.format(max_display, len(choices))
        choices = choices[:max_display] + [ellipsis]
    return sep.join((str(c) for c in choices))