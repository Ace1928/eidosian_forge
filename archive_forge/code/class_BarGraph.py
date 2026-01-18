from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
class BarGraph(Widget, metaclass=BarGraphMeta):
    _sizing = frozenset([Sizing.BOX])
    ignore_focus = True
    eighths = BAR_SYMBOLS.VERTICAL[:8]
    hlines = '_⎺⎻─⎼⎽'

    def __init__(self, attlist, hatt=None, satt=None) -> None:
        """
        Create a bar graph with the passed display characteristics.
        see set_segment_attributes for a description of the parameters.
        """
        super().__init__()
        self.set_segment_attributes(attlist, hatt, satt)
        self.set_data([], 1, None)
        self.set_bar_width(None)

    def set_segment_attributes(self, attlist, hatt=None, satt=None):
        """
        :param attlist: list containing display attribute or
                        (display attribute, character) tuple for background,
                        first segment, and optionally following segments.
                        ie. len(attlist) == num segments+1
                        character defaults to ' ' if not specified.
        :param hatt: list containing attributes for horizontal lines. First
                     element is for lines on background, second is for lines
                     on first segment, third is for lines on second segment
                     etc.
        :param satt: dictionary containing attributes for smoothed
                     transitions of bars in UTF-8 display mode. The values
                     are in the form:

                       (fg,bg) : attr

                     fg and bg are integers where 0 is the graph background,
                     1 is the first segment, 2 is the second, ...
                     fg > bg in all values.  attr is an attribute with a
                     foreground corresponding to fg and a background
                     corresponding to bg.

        If satt is not None and the bar graph is being displayed in
        a terminal using the UTF-8 encoding then the character cell
        that is shared between the segments specified will be smoothed
        with using the UTF-8 vertical eighth characters.

        eg: set_segment_attributes( ['no', ('unsure',"?"), 'yes'] )
        will use the attribute 'no' for the background (the area from
        the top of the graph to the top of the bar), question marks
        with the attribute 'unsure' will be used for the topmost
        segment of the bar, and the attribute 'yes' will be used for
        the bottom segment of the bar.
        """
        self.attr = []
        self.char = []
        if len(attlist) < 2:
            raise BarGraphError(f'attlist must include at least background and seg1: {attlist!r}')
        if len(attlist) < 2:
            raise BarGraphError('must at least specify bg and fg!')
        for a in attlist:
            if not isinstance(a, tuple):
                self.attr.append(a)
                self.char.append(' ')
            else:
                attr, ch = a
                self.attr.append(attr)
                self.char.append(ch)
        self.hatt = []
        if hatt is None:
            hatt = [self.attr[0]]
        elif not isinstance(hatt, list):
            hatt = [hatt]
        self.hatt = hatt
        if satt is None:
            satt = {}
        for i in satt.items():
            try:
                (fg, bg), attr = i
            except ValueError as exc:
                raise BarGraphError(f'satt not in (fg,bg:attr) form: {i!r}').with_traceback(exc.__traceback__) from exc
            if not isinstance(fg, int) or fg >= len(attlist):
                raise BarGraphError(f'fg not valid integer: {fg!r}')
            if not isinstance(bg, int) or bg >= len(attlist):
                raise BarGraphError(f'bg not valid integer: {fg!r}')
            if fg <= bg:
                raise BarGraphError(f'fg ({fg}) not > bg ({bg})')
        self.satt = satt

    def set_data(self, bardata, top: float, hlines=None) -> None:
        """
        Store bar data, bargraph top and horizontal line positions.

        bardata -- a list of bar values.
        top -- maximum value for segments within bardata
        hlines -- None or a bar value marking horizontal line positions

        bar values are [ segment1, segment2, ... ] lists where top is
        the maximal value corresponding to the top of the bar graph and
        segment1, segment2, ... are the values for the top of each
        segment of this bar.  Simple bar graphs will only have one
        segment in each bar value.

        Eg: if top is 100 and there is a bar value of [ 80, 30 ] then
        the top of this bar will be at 80% of full height of the graph
        and it will have a second segment that starts at 30%.
        """
        if hlines is not None:
            hlines = sorted(hlines[:], reverse=True)
        self.data = (bardata, top, hlines)
        self._invalidate()

    def _get_data(self, size: tuple[int, int]):
        """
        Return (bardata, top, hlines)

        This function is called by render to retrieve the data for
        the graph. It may be overloaded to create a dynamic bar graph.

        This implementation will truncate the bardata list returned
        if not all bars will fit within maxcol.
        """
        maxcol, maxrow = size
        bardata, top, hlines = self.data
        widths = self.calculate_bar_widths((maxcol, maxrow), bardata)
        if len(bardata) > len(widths):
            return (bardata[:len(widths)], top, hlines)
        return (bardata, top, hlines)

    def set_bar_width(self, width: int | None):
        """
        Set a preferred bar width for calculate_bar_widths to use.

        width -- width of bar or None for automatic width adjustment
        """
        if width is not None and width <= 0:
            raise ValueError(width)
        self.bar_width = width
        self._invalidate()

    def calculate_bar_widths(self, size: tuple[int, int], bardata):
        """
        Return a list of bar widths, one for each bar in data.

        If self.bar_width is None this implementation will stretch
        the bars across the available space specified by maxcol.
        """
        maxcol, _maxrow = size
        if self.bar_width is not None:
            return [self.bar_width] * min(len(bardata), maxcol // self.bar_width)
        if len(bardata) >= maxcol:
            return [1] * maxcol
        widths = []
        grow = maxcol
        remain = len(bardata)
        for _row in bardata:
            w = int(float(grow) / remain + 0.5)
            widths.append(w)
            grow -= w
            remain -= 1
        return widths

    def selectable(self) -> Literal[False]:
        """
        Return False.
        """
        return False

    def use_smoothed(self) -> bool:
        return self.satt and get_encoding_mode() == 'utf8'

    def calculate_display(self, size: tuple[int, int]):
        """
        Calculate display data.
        """
        maxcol, maxrow = size
        bardata, top, hlines = self.get_data((maxcol, maxrow))
        widths = self.calculate_bar_widths((maxcol, maxrow), bardata)
        if self.use_smoothed():
            disp = calculate_bargraph_display(bardata, top, widths, maxrow * 8)
            disp = self.smooth_display(disp)
        else:
            disp = calculate_bargraph_display(bardata, top, widths, maxrow)
        if hlines:
            disp = self.hlines_display(disp, top, hlines, maxrow)
        return disp

    def hlines_display(self, disp, top: int, hlines, maxrow: int):
        """
        Add hlines to display structure represented as bar_type tuple
        values:
        (bg, 0-5)
        bg is the segment that has the hline on it
        0-5 is the hline graphic to use where 0 is a regular underscore
        and 1-5 are the UTF-8 horizontal scan line characters.
        """
        if self.use_smoothed():
            shiftr = 0
            r = [(0.2, 1), (0.4, 2), (0.6, 3), (0.8, 4), (1.0, 5)]
        else:
            shiftr = 0.5
            r = [(1.0, 0)]
        rhl = []
        for h in hlines:
            rh = float(top - h) * maxrow / top - shiftr
            if rh < 0:
                continue
            rhl.append(rh)
        hrows = []
        last_i = -1
        for rh in rhl:
            i = int(rh)
            if i == last_i:
                continue
            f = rh - i
            for spl, chnum in r:
                if f < spl:
                    hrows.append((i, chnum))
                    break
            last_i = i

        def fill_row(row, chnum):
            rout = []
            for bar_type, width in row:
                if isinstance(bar_type, int) and len(self.hatt) > bar_type:
                    rout.append(((bar_type, chnum), width))
                    continue
                rout.append((bar_type, width))
            return rout
        o = []
        k = 0
        rnum = 0
        for y_count, row in disp:
            if k >= len(hrows):
                o.append((y_count, row))
                continue
            end_block = rnum + y_count
            while k < len(hrows) and hrows[k][0] < end_block:
                i, chnum = hrows[k]
                if i - rnum > 0:
                    o.append((i - rnum, row))
                o.append((1, fill_row(row, chnum)))
                rnum = i + 1
                k += 1
            if rnum < end_block:
                o.append((end_block - rnum, row))
                rnum = end_block
        return o

    def smooth_display(self, disp):
        """
        smooth (col, row*8) display into (col, row) display using
        UTF vertical eighth characters represented as bar_type
        tuple values:
        ( fg, bg, 1-7 )
        where fg is the lower segment, bg is the upper segment and
        1-7 is the vertical eighth character to use.
        """
        o = []
        r = 0

        def seg_combine(a, b):
            (bt1, w1), (bt2, w2) = (a, b)
            if (bt1, w1) == (bt2, w2):
                return ((bt1, w1), None, None)
            wmin = min(w1, w2)
            l1 = l2 = None
            if w1 > w2:
                l1 = (bt1, w1 - w2)
            elif w2 > w1:
                l2 = (bt2, w2 - w1)
            if isinstance(bt1, tuple):
                return ((bt1, wmin), l1, l2)
            if (bt2, bt1) not in self.satt:
                if r < 4:
                    return ((bt2, wmin), l1, l2)
                return ((bt1, wmin), l1, l2)
            return (((bt2, bt1, 8 - r), wmin), l1, l2)

        def row_combine_last(count: int, row):
            o_count, o_row = o[-1]
            row = row[:]
            o_row = o_row[:]
            widget_list = []
            while row:
                (bt, w), l1, l2 = seg_combine(o_row.pop(0), row.pop(0))
                if widget_list and widget_list[-1][0] == bt:
                    widget_list[-1] = (bt, widget_list[-1][1] + w)
                else:
                    widget_list.append((bt, w))
                if l1:
                    o_row = [l1, *o_row]
                if l2:
                    row = [l2, *row]
            if o_row:
                raise BarGraphError(o_row)
            o[-1] = (o_count + count, widget_list)
        for y_count, row in disp:
            if r:
                count = min(8 - r, y_count)
                row_combine_last(count, row)
                y_count -= count
                r += count
                r = r % 8
                if not y_count:
                    continue
            if r != 0:
                raise BarGraphError
            if y_count > 7:
                o.append((y_count // 8 * 8, row))
                y_count %= 8
                if not y_count:
                    continue
            o.append((y_count, row))
            r = y_count
        return [(y // 8, row) for y, row in o]

    def render(self, size: tuple[int, int], focus: bool=False) -> CompositeCanvas:
        """
        Render BarGraph.
        """
        maxcol, maxrow = size
        disp = self.calculate_display((maxcol, maxrow))
        combinelist = []
        for y_count, row in disp:
            widget_list = []
            for bar_type, width in row:
                if isinstance(bar_type, tuple):
                    if len(bar_type) == 3:
                        fg, bg, k = bar_type
                        a = self.satt[fg, bg]
                        t = self.eighths[k] * width
                    else:
                        bg, k = bar_type
                        a = self.hatt[bg]
                        t = self.hlines[k] * width
                else:
                    a = self.attr[bar_type]
                    t = self.char[bar_type] * width
                widget_list.append((a, t))
            c = Text(widget_list).render((maxcol,))
            if c.rows() != 1:
                raise BarGraphError('Invalid characters in BarGraph!')
            combinelist += [(c, None, False)] * y_count
        canv = CanvasCombine(combinelist)
        return canv