from __future__ import annotations
import functools
import typing
from urwid.str_util import calc_text_pos, calc_width, get_char_width, is_wide_char, move_next_char, move_prev_char
from urwid.util import calc_trim_text, get_encoding
class StandardTextLayout(TextLayout):

    def __init__(self) -> None:
        pass

    def supports_align_mode(self, align: Literal['left', 'center', 'right'] | Align) -> bool:
        """Return True if align is 'left', 'center' or 'right'."""
        return align in {'left', 'center', 'right'}

    def supports_wrap_mode(self, wrap: Literal['any', 'space', 'clip', 'ellipsis'] | WrapMode) -> bool:
        """Return True if wrap is 'any', 'space', 'clip' or 'ellipsis'."""
        return wrap in {'any', 'space', 'clip', 'ellipsis'}

    def layout(self, text: str | bytes, width: int, align: Literal['left', 'center', 'right'] | Align, wrap: Literal['any', 'space', 'clip', 'ellipsis'] | WrapMode) -> list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]]:
        """Return a layout structure for text."""
        try:
            segs = self.calculate_text_segments(text, width, wrap)
            return self.align_layout(text, width, segs, wrap, align)
        except CanNotDisplayText:
            return [[]]

    def pack(self, maxcol: int, layout: list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]]) -> int:
        """Return a minimal maxcol value that would result in the same number of lines for layout.

        layout must be a layout structure returned by self.layout().
        """
        maxwidth = 0
        if not layout:
            raise ValueError(f'huh? empty layout?: {layout!r}')
        for lines in layout:
            lw = line_width(lines)
            if lw >= maxcol:
                return maxcol
            maxwidth = max(maxwidth, lw)
        return maxwidth

    def align_layout(self, text: str | bytes, width: int, segs: list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]], wrap: Literal['any', 'space', 'clip', 'ellipsis'] | WrapMode, align: Literal['left', 'center', 'right'] | Align) -> list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]]:
        """Convert the layout segments to an aligned layout."""
        out = []
        for lines in segs:
            sc = line_width(lines)
            if sc == width or align == 'left':
                out.append(lines)
                continue
            if align == 'right':
                out.append([(width - sc, None), *lines])
                continue
            if align != 'center':
                raise ValueError(align)
            pad_trim_left = (width - sc + 1) // 2
            out.append([(pad_trim_left, None), *lines] if pad_trim_left else lines)
        return out

    def _calculate_trimmed_segments(self, text: str | bytes, width: int, wrap: Literal['any', 'space', 'clip', 'ellipsis'] | WrapMode) -> list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]]:
        """Calculate text segments for cases of a text trimmed (wrap is clip or ellipsis)."""
        segments = []
        nl: str | bytes = '\n' if isinstance(text, str) else b'\n'
        encoding = get_encoding()
        ellipsis_string = get_ellipsis_string(encoding)
        ellipsis_width = _get_width(ellipsis_string)
        while width - 1 < ellipsis_width and ellipsis_string:
            ellipsis_string = ellipsis_string[:-1]
            ellipsis_width = _get_width(ellipsis_string)
        ellipsis_char = ellipsis_string.encode(encoding)
        idx = 0
        while idx <= len(text):
            nl_pos = text.find(nl, idx)
            if nl_pos == -1:
                nl_pos = len(text)
            screen_columns = calc_width(text, idx, nl_pos)
            if wrap == 'ellipsis' and screen_columns > width and ellipsis_width:
                trimmed = True
                start_off, end_off, pad_left, pad_right = calc_trim_text(text, idx, nl_pos, 0, width - ellipsis_width)
                if pad_left != 0:
                    raise ValueError(f'Invalid padding for start column==0: {pad_left!r}')
                if start_off != idx:
                    raise ValueError(f'Invalid start offset for  start column==0 and position={idx!r}: {start_off!r}')
                screen_columns = width - 1 - pad_right
            else:
                trimmed = False
                end_off = nl_pos
                pad_right = 0
            line = []
            if idx != end_off:
                line += [(screen_columns, idx, end_off)]
            if trimmed:
                line += [(ellipsis_width, end_off, ellipsis_char)]
            line += [(pad_right, end_off)]
            segments.append(line)
            idx = nl_pos + 1
        return segments

    def calculate_text_segments(self, text: str | bytes, width: int, wrap: Literal['any', 'space', 'clip', 'ellipsis'] | WrapMode) -> list[list[tuple[int, int, int | bytes] | tuple[int, int | None]]]:
        """
        Calculate the segments of text to display given width screen columns to display them.

        text - unicode text or byte string to display
        width - number of available screen columns
        wrap - wrapping mode used

        Returns a layout structure without an alignment applied.
        """
        if wrap in {'clip', 'ellipsis'}:
            return self._calculate_trimmed_segments(text, width, wrap)
        nl, nl_o, sp_o = ('\n', '\n', ' ')
        if isinstance(text, bytes):
            nl = b'\n'
            nl_o = ord(nl_o)
            sp_o = ord(sp_o)
        segments = []
        idx = 0
        while idx <= len(text):
            nl_pos = text.find(nl, idx)
            if nl_pos == -1:
                nl_pos = len(text)
            screen_columns = calc_width(text, idx, nl_pos)
            if screen_columns == 0:
                segments.append([(0, nl_pos)])
                idx = nl_pos + 1
                continue
            if screen_columns <= width:
                segments.append([(screen_columns, idx, nl_pos), (0, nl_pos)])
                idx = nl_pos + 1
                continue
            pos, screen_columns = calc_text_pos(text, idx, nl_pos, width)
            if pos == idx:
                raise CanNotDisplayText('Wide character will not fit in 1-column width')
            if wrap == 'any':
                segments.append([(screen_columns, idx, pos)])
                idx = pos
                continue
            if wrap != 'space':
                raise ValueError(wrap)
            if text[pos] == sp_o:
                segments.append([(screen_columns, idx, pos), (0, pos)])
                idx = pos + 1
                continue
            if is_wide_char(text, pos):
                segments.append([(screen_columns, idx, pos)])
                idx = pos
                continue
            prev = pos
            while prev > idx:
                prev = move_prev_char(text, idx, prev)
                if text[prev] == sp_o:
                    screen_columns = calc_width(text, idx, prev)
                    line = [(0, prev)]
                    if idx != prev:
                        line = [(screen_columns, idx, prev), *line]
                    segments.append(line)
                    idx = prev + 1
                    break
                if is_wide_char(text, prev):
                    next_char = move_next_char(text, prev, pos)
                    screen_columns = calc_width(text, idx, next_char)
                    segments.append([(screen_columns, idx, next_char)])
                    idx = next_char
                    break
            else:
                if segments and (len(segments[-1]) == 2 or (len(segments[-1]) == 1 and len(segments[-1][0]) == 2)):
                    if len(segments[-1]) == 1:
                        [(h_sc, h_off)] = segments[-1]
                        p_sc = 0
                        p_off = _p_end = h_off
                    else:
                        [(p_sc, p_off, _p_end), (h_sc, h_off)] = segments[-1]
                    if p_sc < width and h_sc == 0 and (text[h_off] == sp_o):
                        del segments[-1]
                        idx = p_off
                        pos, screen_columns = calc_text_pos(text, idx, nl_pos, width)
                        segments.append([(screen_columns, idx, pos)])
                        idx = pos
                        if idx < len(text) and text[idx] in {sp_o, nl_o}:
                            segments[-1].append((0, idx))
                            idx += 1
                        continue
                segments.append([(screen_columns, idx, pos)])
                idx = pos
        return segments