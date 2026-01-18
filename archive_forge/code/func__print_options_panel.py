import inspect
import sys
from collections import defaultdict
from gettext import gettext as _
from os import getenv
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Union
import click
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, RenderableType, group
from rich.emoji import Emoji
from rich.highlighter import RegexHighlighter
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
def _print_options_panel(*, name: str, params: Union[List[click.Option], List[click.Argument]], ctx: click.Context, markup_mode: MarkupMode, console: Console) -> None:
    options_rows: List[List[RenderableType]] = []
    required_rows: List[Union[str, Text]] = []
    for param in params:
        opt_long_strs = []
        opt_short_strs = []
        secondary_opt_long_strs = []
        secondary_opt_short_strs = []
        for opt_str in param.opts:
            if '--' in opt_str:
                opt_long_strs.append(opt_str)
            else:
                opt_short_strs.append(opt_str)
        for opt_str in param.secondary_opts:
            if '--' in opt_str:
                secondary_opt_long_strs.append(opt_str)
            else:
                secondary_opt_short_strs.append(opt_str)
        metavar = Text(style=STYLE_METAVAR, overflow='fold')
        metavar_str = param.make_metavar()
        if isinstance(param, click.Argument) and param.name and (metavar_str == param.name.upper()):
            metavar_str = param.type.name.upper()
        if metavar_str != 'BOOLEAN':
            metavar.append(metavar_str)
        if isinstance(param.type, click.types._NumberRangeBase) and isinstance(param, click.Option) and (not (param.count and param.type.min == 0 and (param.type.max is None))):
            range_str = param.type._describe_range()
            if range_str:
                metavar.append(RANGE_STRING.format(range_str))
        required: Union[str, Text] = ''
        if param.required:
            required = Text(REQUIRED_SHORT_STRING, style=STYLE_REQUIRED_SHORT)

        class MetavarHighlighter(RegexHighlighter):
            highlights = ['^(?P<metavar_sep>(\\[|<))', '(?P<metavar_sep>\\|)', '(?P<metavar_sep>(\\]|>)$)']
        metavar_highlighter = MetavarHighlighter()
        required_rows.append(required)
        options_rows.append([highlighter(','.join(opt_long_strs)), highlighter(','.join(opt_short_strs)), negative_highlighter(','.join(secondary_opt_long_strs)), negative_highlighter(','.join(secondary_opt_short_strs)), metavar_highlighter(metavar), _get_parameter_help(param=param, ctx=ctx, markup_mode=markup_mode)])
    rows_with_required: List[List[RenderableType]] = []
    if any(required_rows):
        for required, row in zip(required_rows, options_rows):
            rows_with_required.append([required, *row])
    else:
        rows_with_required = options_rows
    if options_rows:
        t_styles: Dict[str, Any] = {'show_lines': STYLE_OPTIONS_TABLE_SHOW_LINES, 'leading': STYLE_OPTIONS_TABLE_LEADING, 'box': STYLE_OPTIONS_TABLE_BOX, 'border_style': STYLE_OPTIONS_TABLE_BORDER_STYLE, 'row_styles': STYLE_OPTIONS_TABLE_ROW_STYLES, 'pad_edge': STYLE_OPTIONS_TABLE_PAD_EDGE, 'padding': STYLE_OPTIONS_TABLE_PADDING}
        box_style = getattr(box, t_styles.pop('box'), None)
        options_table = Table(highlight=True, show_header=False, expand=True, box=box_style, **t_styles)
        for row in rows_with_required:
            options_table.add_row(*row)
        console.print(Panel(options_table, border_style=STYLE_OPTIONS_PANEL_BORDER, title=name, title_align=ALIGN_OPTIONS_PANEL))