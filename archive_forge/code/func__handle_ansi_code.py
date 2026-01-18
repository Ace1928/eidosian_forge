import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def _handle_ansi_code(self, ansi: str, styles_used: Set[str], state: _State) -> Iterator[Union[str, CursorMoveUp]]:
    last_end = 0
    for match in self.ansi_codes_prog.finditer(ansi):
        yield ansi[last_end:match.start()]
        last_end = match.end()
        params: Union[str, List[int]]
        params, command = match.groups()
        if command not in 'mMA':
            continue
        if command == 'A':
            yield CursorMoveUp()
            continue
        while True:
            param_len = len(params)
            params = params.replace('::', ':')
            params = params.replace(';;', ';')
            if len(params) == param_len:
                break
        try:
            params = [int(x) for x in re.split('[;:]', params)]
        except ValueError:
            params = [ANSI_FULL_RESET]
        last_null_index = None
        skip_after_index = -1
        for i, v in enumerate(params):
            if i <= skip_after_index:
                continue
            if v == ANSI_FULL_RESET:
                last_null_index = i
            elif v in (ANSI_FOREGROUND, ANSI_BACKGROUND):
                try:
                    x_bit_color_id = params[i + 1]
                except IndexError:
                    x_bit_color_id = -1
                is_256_color = x_bit_color_id == ANSI_256_COLOR_ID
                shift = 2 if is_256_color else 4
                skip_after_index = i + shift
        if last_null_index is not None:
            params = params[last_null_index + 1:]
            if state.inside_span:
                state.inside_span = False
                if self.latex:
                    yield '}'
                else:
                    yield '</span>'
            state.reset()
            if not params:
                continue
        skip_after_index = -1
        for i, v in enumerate(params):
            if i <= skip_after_index:
                continue
            is_x_bit_color = v in (ANSI_FOREGROUND, ANSI_BACKGROUND)
            try:
                x_bit_color_id = params[i + 1]
            except IndexError:
                x_bit_color_id = -1
            is_256_color = x_bit_color_id == ANSI_256_COLOR_ID
            is_truecolor = x_bit_color_id == ANSI_TRUECOLOR_ID
            if is_x_bit_color and is_256_color:
                try:
                    parameter: Optional[str] = str(params[i + 2])
                except IndexError:
                    continue
                skip_after_index = i + 2
            elif is_x_bit_color and is_truecolor:
                try:
                    state.adjust_truecolor(v, params[i + 2], params[i + 3], params[i + 4])
                except IndexError:
                    continue
                skip_after_index = i + 4
                continue
            else:
                parameter = None
            state.adjust(v, parameter=parameter)
        if state.inside_span:
            if self.latex:
                yield '}'
            else:
                yield '</span>'
            state.inside_span = False
        css_classes = state.to_css_classes()
        if not css_classes:
            continue
        styles_used.update(css_classes)
        if self.inline:
            self.styles.update(pop_truecolor_styles())
            if self.latex:
                style = [self.styles[klass].kwl[0][1] for klass in css_classes if self.styles[klass].kwl[0][0] == 'color']
                yield ('\\textcolor[HTML]{%s}{' % style[0])
            else:
                style = [self.styles[klass].kw for klass in css_classes if klass in self.styles]
                yield ('<span style="%s">' % '; '.join(style))
        elif self.latex:
            yield ('\\textcolor{%s}{' % ' '.join(css_classes))
        else:
            yield ('<span class="%s">' % ' '.join(css_classes))
        state.inside_span = True
    yield ansi[last_end:]