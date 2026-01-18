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
def _get_parameter_help(*, param: Union[click.Option, click.Argument, click.Parameter], ctx: click.Context, markup_mode: MarkupMode) -> Columns:
    """Build primary help text for a click option or argument.

    Returns the prose help text for an option or argument, rendered either
    as a Rich Text object or as Markdown.
    Additional elements are appended to show the default and required status if
    applicable.
    """
    from .core import TyperArgument, TyperOption
    items: List[Union[Text, Markdown]] = []
    envvar = getattr(param, 'envvar', None)
    var_str = ''
    if envvar is None:
        if getattr(param, 'allow_from_autoenv', None) and getattr(ctx, 'auto_envvar_prefix', None) is not None and (param.name is not None):
            envvar = f'{ctx.auto_envvar_prefix}_{param.name.upper()}'
    if envvar is not None:
        var_str = envvar if isinstance(envvar, str) else ', '.join((str(d) for d in envvar))
    help_value: Union[str, None] = getattr(param, 'help', None)
    if help_value:
        paragraphs = help_value.split('\n\n')
        if markup_mode != MARKUP_MODE_MARKDOWN:
            paragraphs = [x.replace('\n', ' ').strip() if not x.startswith('\x08') else '{}\n'.format(x.strip('\x08\n')) for x in paragraphs]
        items.append(_make_rich_rext(text='\n'.join(paragraphs).strip(), style=STYLE_OPTION_HELP, markup_mode=markup_mode))
    if envvar and getattr(param, 'show_envvar', None):
        items.append(Text(ENVVAR_STRING.format(var_str), style=STYLE_OPTION_ENVVAR))
    if isinstance(param, (TyperOption, TyperArgument)):
        if param.show_default:
            show_default_is_str = isinstance(param.show_default, str)
            default_value = param._extract_default_help_str(ctx=ctx)
            default_str = param._get_default_string(ctx=ctx, show_default_is_str=show_default_is_str, default_value=default_value)
            if default_str:
                items.append(Text(DEFAULT_STRING.format(default_str), style=STYLE_OPTION_DEFAULT))
    if param.required:
        items.append(Text(REQUIRED_LONG_STRING, style=STYLE_REQUIRED_LONG))
    return Columns(items)