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
class MyOptionParser(argparse.ArgumentParser):

    def __init__(self, parser: Parser, extra_info: Optional[Dict[str, Any]]=None, prog: Optional[str]=None) -> None:
        self._parser = parser
        super().__init__(prog=prog, usage=parser._usage, add_help=False, formatter_class=DropShorterLongHelpFormatter, allow_abbrev=False)
        self.extra_info = extra_info if extra_info else {}

    def error(self, message: str) -> NoReturn:
        """Transform argparse error message into UsageError."""
        msg = f'{self.prog}: error: {message}'
        if hasattr(self._parser, '_config_source_hint'):
            msg = f'{msg} ({self._parser._config_source_hint})'
        raise UsageError(self.format_usage() + msg)

    def parse_args(self, args: Optional[Sequence[str]]=None, namespace: Optional[argparse.Namespace]=None) -> argparse.Namespace:
        """Allow splitting of positional arguments."""
        parsed, unrecognized = self.parse_known_args(args, namespace)
        if unrecognized:
            for arg in unrecognized:
                if arg and arg[0] == '-':
                    lines = ['unrecognized arguments: %s' % ' '.join(unrecognized)]
                    for k, v in sorted(self.extra_info.items()):
                        lines.append(f'  {k}: {v}')
                    self.error('\n'.join(lines))
            getattr(parsed, FILE_OR_DIR).extend(unrecognized)
        return parsed
    if sys.version_info[:2] < (3, 9):

        def _parse_optional(self, arg_string: str) -> Optional[Tuple[Optional[argparse.Action], str, Optional[str]]]:
            if not arg_string:
                return None
            if arg_string[0] not in self.prefix_chars:
                return None
            if arg_string in self._option_string_actions:
                action = self._option_string_actions[arg_string]
                return (action, arg_string, None)
            if len(arg_string) == 1:
                return None
            if '=' in arg_string:
                option_string, explicit_arg = arg_string.split('=', 1)
                if option_string in self._option_string_actions:
                    action = self._option_string_actions[option_string]
                    return (action, option_string, explicit_arg)
            if self.allow_abbrev or not arg_string.startswith('--'):
                option_tuples = self._get_option_tuples(arg_string)
                if len(option_tuples) > 1:
                    msg = gettext('ambiguous option: %(option)s could match %(matches)s')
                    options = ', '.join((option for _, option, _ in option_tuples))
                    self.error(msg % {'option': arg_string, 'matches': options})
                elif len(option_tuples) == 1:
                    option_tuple, = option_tuples
                    return option_tuple
            if self._negative_number_matcher.match(arg_string):
                if not self._has_negative_number_optionals:
                    return None
            if ' ' in arg_string:
                return None
            return (None, arg_string, None)