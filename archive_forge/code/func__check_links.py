import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def _check_links(parts: List[Union[str, OSC_Link]]) -> Iterator[str]:
    for part in parts:
        if isinstance(part, str):
            if self.linkify:
                yield self.do_linkify(part)
            else:
                yield part
        elif isinstance(part, OSC_Link):
            yield self.handle_osc_links(part)
        else:
            yield part