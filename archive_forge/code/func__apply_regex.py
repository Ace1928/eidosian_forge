import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def _apply_regex(self, ansi: str, styles_used: Set[str]) -> Iterator[Union[str, OSC_Link, CursorMoveUp]]:
    if self.escaped:
        if self.latex:
            specials = OrderedDict([])
        else:
            specials = OrderedDict([('&', '&amp;'), ('<', '&lt;'), ('>', '&gt;')])
        for pattern, special in specials.items():
            ansi = ansi.replace(pattern, special)

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
    ansi = ''.join(_vt100_box_drawing())

    def _osc_link(ansi: str) -> Iterator[Union[str, OSC_Link]]:
        last_end = 0
        for match in self.osc_link_re.finditer(ansi):
            trailer = ansi[last_end:match.start()]
            yield trailer
            url = match.groups()[0]
            text = match.groups()[1]
            yield OSC_Link(url, text)
            last_end = match.end()
        yield ansi[last_end:]
    state = _State()
    for part in _osc_link(ansi):
        if isinstance(part, OSC_Link):
            yield part
        else:
            yield from self._handle_ansi_code(part, styles_used, state)
    if state.inside_span:
        if self.latex:
            yield '}'
        else:
            yield '</span>'