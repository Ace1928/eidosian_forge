from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.six import text_type
Quotes a value for PowerShell.

    Quotes a value to be safely used by a PowerShell expression. The input
    string because something that is safely wrapped in single quotes.

    Args:
        s: The string to quote.

    Returns:
        (text_type): The quoted string value.
    