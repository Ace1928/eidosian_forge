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
def _report_sort_key(report_key: str) -> Tuple[str, str]:
    """
    Sorting name for reports.

    Sorts by main metric alphabetically, then by task.
    """
    fields = report_key.split('/')
    main_key = fields.pop(-1)
    sub_key = '/'.join(fields)
    return (sub_key or 'all', main_key)