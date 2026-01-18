from __future__ import annotations
import argparse
import ast
import logging
from typing import Any
from typing import Generator
import pyflakes.checker
from flake8.options.manager import OptionManager
class FlakesChecker(pyflakes.checker.Checker):
    """Subclass the Pyflakes checker to conform with the flake8 API."""
    with_doctest = False

    def __init__(self, tree: ast.AST, filename: str) -> None:
        """Initialize the PyFlakes plugin with an AST tree and filename."""
        super().__init__(tree, filename=filename, withDoctest=self.with_doctest)

    @classmethod
    def add_options(cls, parser: OptionManager) -> None:
        """Register options for PyFlakes on the Flake8 OptionManager."""
        parser.add_option('--builtins', parse_from_config=True, comma_separated_list=True, help='define more built-ins, comma separated')
        parser.add_option('--doctests', default=False, action='store_true', parse_from_config=True, help='also check syntax of the doctests')

    @classmethod
    def parse_options(cls, options: argparse.Namespace) -> None:
        """Parse option values from Flake8's OptionManager."""
        if options.builtins:
            cls.builtIns = cls.builtIns.union(options.builtins)
        cls.with_doctest = options.doctests

    def run(self) -> Generator[tuple[int, int, str, type[Any]], None, None]:
        """Run the plugin."""
        for message in self.messages:
            col = getattr(message, 'col', 0)
            yield (message.lineno, col, '{} {}'.format(FLAKE8_PYFLAKES_CODES.get(type(message).__name__, 'F999'), message.message % message.message_args), message.__class__)