import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
class Ansi2HTMLConverter:
    """Convert Ansi color codes to CSS+HTML

    Example:

    >>> conv = Ansi2HTMLConverter()
    >>> ansi = " ".join(sys.stdin.readlines())
    >>> html = conv.convert(ansi)
    """

    def __init__(self, latex: bool=False, inline: bool=False, dark_bg: bool=True, line_wrap: bool=True, font_size: str='normal', linkify: bool=False, escaped: bool=True, markup_lines: bool=False, output_encoding: str='utf-8', scheme: str='ansi2html', title: str='') -> None:
        self.latex = latex
        self.inline = inline
        self.dark_bg = dark_bg
        self.line_wrap = line_wrap
        self.font_size = font_size
        self.linkify = linkify
        self.escaped = escaped
        self.markup_lines = markup_lines
        self.output_encoding = output_encoding
        self.scheme = scheme
        self.title = title
        self._attrs: Attributes
        self.hyperref = False
        if inline:
            self.styles = dict([(item.klass.strip('.'), item) for item in get_styles(self.dark_bg, self.line_wrap, self.scheme)])
        self.vt100_box_codes_prog = re.compile('\x1b\\(([B0])')
        self.ansi_codes_prog = re.compile('\x1b\\[([\\d;:]*)([a-zA-z])')
        self.url_matcher = re.compile("(((((https?|ftps?|gopher|telnet|nntp)://)|(mailto:|news:))(%[0-9A-Fa-f]{2}|[-()_.!~*\\';/?#:@&=+$,A-Za-z0-9])+)([).!\\';/?:,][\\s])?)")
        self.osc_link_re = re.compile('\x1b\\]8;;(.*?)\x07(.*?)\x1b\\]8;;\x07')

    def do_linkify(self, line: str) -> str:
        if not isinstance(line, str):
            return line
        if self.latex:
            return self.url_matcher.sub('\\\\url{\\1}', line)
        return self.url_matcher.sub('<a href="\\1">\\1</a>', line)

    def handle_osc_links(self, part: OSC_Link) -> str:
        if self.latex:
            self.hyperref = True
            return '\\href{%s}{%s}' % (part.url, part.text)
        return '<a href="%s">%s</a>' % (part.url, part.text)

    def apply_regex(self, ansi: str) -> Tuple[str, Set[str]]:
        styles_used: Set[str] = set()
        all_parts = self._apply_regex(ansi, styles_used)
        no_cursor_parts = self._collapse_cursor(all_parts)
        no_cursor_parts = list(no_cursor_parts)

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
        parts = list(_check_links(no_cursor_parts))
        combined = ''.join(parts)
        if self.markup_lines and (not self.latex):
            combined = '\n'.join(['<span id="line-%i">%s</span>' % (i, line) for i, line in enumerate(combined.split('\n'))])
        return (combined, styles_used)

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

    def _collapse_cursor(self, parts: Iterator[Union[str, OSC_Link, CursorMoveUp]]) -> List[Union[str, OSC_Link]]:
        """Act on any CursorMoveUp commands by deleting preceding tokens"""
        final_parts: List[Union[str, OSC_Link]] = []
        for part in parts:
            if not part:
                continue
            if isinstance(part, CursorMoveUp):
                if final_parts:
                    final_parts.pop()
                while final_parts and (isinstance(final_parts[-1], OSC_Link) or (isinstance(final_parts[-1], str) and '\n' not in final_parts[-1])):
                    final_parts.pop()
                continue
            final_parts.append(part)
        return final_parts

    def prepare(self, ansi: str='', ensure_trailing_newline: bool=False) -> Attributes:
        """Load the contents of 'ansi' into this object"""
        body, styles = self.apply_regex(ansi)
        if ensure_trailing_newline and _needs_extra_newline(body):
            body += '\n'
        self._attrs = {'dark_bg': self.dark_bg, 'line_wrap': self.line_wrap, 'font_size': self.font_size, 'body': body, 'styles': styles}
        return self._attrs

    def convert(self, ansi: str, full: bool=True, ensure_trailing_newline: bool=False) -> str:
        """
        :param ansi: ANSI sequence to convert.
        :param full: Whether to include the full HTML document or only the body.
        :param ensure_trailing_newline: Ensures that ``\\n`` character is present at the end of the output.
        """
        attrs = self.prepare(ansi, ensure_trailing_newline=ensure_trailing_newline)
        if not full:
            return attrs['body']
        if self.latex:
            _template = _latex_template
        else:
            _template = _html_template
        all_styles = get_styles(self.dark_bg, self.line_wrap, self.scheme)
        backgrounds = all_styles[:5]
        used_styles = filter(lambda e: e.klass.lstrip('.') in attrs['styles'], all_styles)
        return _template % {'style': '\n'.join(list(map(str, backgrounds + list(used_styles)))), 'title': self.title, 'font_size': self.font_size, 'content': attrs['body'], 'output_encoding': self.output_encoding, 'hyperref': '\\usepackage{hyperref}' if self.hyperref else ''}

    def produce_headers(self) -> str:
        return '<style type="text/css">\n%(style)s\n</style>\n' % {'style': '\n'.join(map(str, get_styles(self.dark_bg, self.line_wrap, self.scheme)))}