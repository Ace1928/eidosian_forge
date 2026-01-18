import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def _vt100_box_drawing() -> Iterator[str]:
    last_end = 0
    box_drawing_mode = False
    for match in self.vt100_box_codes_prog.finditer(ansi):
        trailer = ansi[last_end:match.start()]
        if box_drawing_mode:
            for char in trailer:
                yield map_vt100_box_code(char)
        else:
            yield trailer
        last_end = match.end()
        box_drawing_mode = match.groups()[0] == '0'
    yield ansi[last_end:]