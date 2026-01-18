from __future__ import annotations
import argparse
import contextlib
import copy
import enum
import functools
import logging
from typing import Generator
from typing import Sequence
from flake8 import defaults
from flake8 import statistics
from flake8 import utils
from flake8.formatting import base as base_formatter
from flake8.violation import Violation
class StyleGuideManager:
    """Manage multiple style guides for a single run."""

    def __init__(self, options: argparse.Namespace, formatter: base_formatter.BaseFormatter, decider: DecisionEngine | None=None) -> None:
        """Initialize our StyleGuide.

        .. todo:: Add parameter documentation.
        """
        self.options = options
        self.formatter = formatter
        self.stats = statistics.Statistics()
        self.decider = decider or DecisionEngine(options)
        self.style_guides: list[StyleGuide] = []
        self.default_style_guide = StyleGuide(options, formatter, self.stats, decider=decider)
        self.style_guides = [self.default_style_guide, *self.populate_style_guides_with(options)]
        self.style_guide_for = functools.lru_cache(maxsize=None)(self._style_guide_for)

    def populate_style_guides_with(self, options: argparse.Namespace) -> Generator[StyleGuide, None, None]:
        """Generate style guides from the per-file-ignores option.

        :param options:
            The original options parsed from the CLI and config file.
        :returns:
            A copy of the default style guide with overridden values.
        """
        per_file = utils.parse_files_to_codes_mapping(options.per_file_ignores)
        for filename, violations in per_file:
            yield self.default_style_guide.copy(filename=filename, extend_ignore_with=violations)

    def _style_guide_for(self, filename: str) -> StyleGuide:
        """Find the StyleGuide for the filename in particular."""
        return max((g for g in self.style_guides if g.applies_to(filename)), key=lambda g: len(g.filename or ''))

    @contextlib.contextmanager
    def processing_file(self, filename: str) -> Generator[StyleGuide, None, None]:
        """Record the fact that we're processing the file's results."""
        guide = self.style_guide_for(filename)
        with guide.processing_file(filename):
            yield guide

    def handle_error(self, code: str, filename: str, line_number: int, column_number: int, text: str, physical_line: str | None=None) -> int:
        """Handle an error reported by a check.

        :param code:
            The error code found, e.g., E123.
        :param filename:
            The file in which the error was found.
        :param line_number:
            The line number (where counting starts at 1) at which the error
            occurs.
        :param column_number:
            The column number (where counting starts at 1) at which the error
            occurs.
        :param text:
            The text of the error message.
        :param physical_line:
            The actual physical line causing the error.
        :returns:
            1 if the error was reported. 0 if it was ignored. This is to allow
            for counting of the number of errors found that were not ignored.
        """
        guide = self.style_guide_for(filename)
        return guide.handle_error(code, filename, line_number, column_number, text, physical_line)