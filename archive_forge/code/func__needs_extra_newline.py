import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def _needs_extra_newline(text: str) -> bool:
    if not text or text.endswith('\n'):
        return False
    return True