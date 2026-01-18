import argparse
from gettext import gettext
import os
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import final
from typing import List
from typing import Literal
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import _pytest._io
from _pytest.config.exceptions import UsageError
from _pytest.deprecated import check_ispytest
class DropShorterLongHelpFormatter(argparse.HelpFormatter):
    """Shorten help for long options that differ only in extra hyphens.

    - Collapse **long** options that are the same except for extra hyphens.
    - Shortcut if there are only two options and one of them is a short one.
    - Cache result on the action object as this is called at least 2 times.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if 'width' not in kwargs:
            kwargs['width'] = _pytest._io.get_terminal_width()
        super().__init__(*args, **kwargs)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        orgstr = super()._format_action_invocation(action)
        if orgstr and orgstr[0] != '-':
            return orgstr
        res: Optional[str] = getattr(action, '_formatted_action_invocation', None)
        if res:
            return res
        options = orgstr.split(', ')
        if len(options) == 2 and (len(options[0]) == 2 or len(options[1]) == 2):
            action._formatted_action_invocation = orgstr
            return orgstr
        return_list = []
        short_long: Dict[str, str] = {}
        for option in options:
            if len(option) == 2 or option[2] == ' ':
                continue
            if not option.startswith('--'):
                raise ArgumentError('long optional argument without "--": [%s]' % option, option)
            xxoption = option[2:]
            shortened = xxoption.replace('-', '')
            if shortened not in short_long or len(short_long[shortened]) < len(xxoption):
                short_long[shortened] = xxoption
        for option in options:
            if len(option) == 2 or option[2] == ' ':
                return_list.append(option)
            if option[2:] == short_long.get(option.replace('-', '')):
                return_list.append(option.replace(' ', '=', 1))
        formatted_action_invocation = ', '.join(return_list)
        action._formatted_action_invocation = formatted_action_invocation
        return formatted_action_invocation

    def _split_lines(self, text, width):
        """Wrap lines after splitting on original newlines.

        This allows to have explicit line breaks in the help text.
        """
        import textwrap
        lines = []
        for line in text.splitlines():
            lines.extend(textwrap.wrap(line.strip(), width))
        return lines