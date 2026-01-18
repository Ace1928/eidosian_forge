from functools import reduce
from datetime import datetime
import re
def _set_time(value):
    if not node.hasAttribute('datatype'):
        dt = _format_test(value)
        if dt != plain:
            node.setAttribute('datatype', dt)
    node.setAttribute('content', value)