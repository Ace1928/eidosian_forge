import configparser
import contextlib
import locale
import logging
import pathlib
import re
import sys
from itertools import chain, groupby, repeat
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union
from pip._vendor.requests.models import Request, Response
from pip._vendor.rich.console import Console, ConsoleOptions, RenderResult
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text
class ExternallyManagedEnvironment(DiagnosticPipError):
    """The current environment is externally managed.

    This is raised when the current environment is externally managed, as
    defined by `PEP 668`_. The ``EXTERNALLY-MANAGED`` configuration is checked
    and displayed when the error is bubbled up to the user.

    :param error: The error message read from ``EXTERNALLY-MANAGED``.
    """
    reference = 'externally-managed-environment'

    def __init__(self, error: Optional[str]) -> None:
        if error is None:
            context = Text(_DEFAULT_EXTERNALLY_MANAGED_ERROR)
        else:
            context = Text(error)
        super().__init__(message='This environment is externally managed', context=context, note_stmt='If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.', hint_stmt=Text('See PEP 668 for the detailed specification.'))

    @staticmethod
    def _iter_externally_managed_error_keys() -> Iterator[str]:
        try:
            category = locale.LC_MESSAGES
        except AttributeError:
            lang: Optional[str] = None
        else:
            lang, _ = locale.getlocale(category)
        if lang is not None:
            yield f'Error-{lang}'
            for sep in ('-', '_'):
                before, found, _ = lang.partition(sep)
                if not found:
                    continue
                yield f'Error-{before}'
        yield 'Error'

    @classmethod
    def from_config(cls, config: Union[pathlib.Path, str]) -> 'ExternallyManagedEnvironment':
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(config, encoding='utf-8')
            section = parser['externally-managed']
            for key in cls._iter_externally_managed_error_keys():
                with contextlib.suppress(KeyError):
                    return cls(section[key])
        except KeyError:
            pass
        except (OSError, UnicodeDecodeError, configparser.ParsingError):
            from pip._internal.utils._log import VERBOSE
            exc_info = logger.isEnabledFor(VERBOSE)
            logger.warning('Failed to read %s', config, exc_info=exc_info)
        return cls(None)