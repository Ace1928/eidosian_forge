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
class DecisionEngine:
    """A class for managing the decision process around violations.

    This contains the logic for whether a violation should be reported or
    ignored.
    """

    def __init__(self, options: argparse.Namespace) -> None:
        """Initialize the engine."""
        self.cache: dict[str, Decision] = {}
        self.selected_explicitly = _explicitly_chosen(option=options.select, extend=options.extend_select)
        self.ignored_explicitly = _explicitly_chosen(option=options.ignore, extend=options.extend_ignore)
        self.selected = _select_ignore(option=options.select, default=(), extended_default=options.extended_default_select, extend=options.extend_select)
        self.ignored = _select_ignore(option=options.ignore, default=defaults.IGNORE, extended_default=options.extended_default_ignore, extend=options.extend_ignore)

    def was_selected(self, code: str) -> Selected | Ignored:
        """Determine if the code has been selected by the user.

        :param code: The code for the check that has been run.
        :returns:
            Selected.Implicitly if the selected list is empty,
            Selected.Explicitly if the selected list is not empty and a match
            was found,
            Ignored.Implicitly if the selected list is not empty but no match
            was found.
        """
        if code.startswith(self.selected_explicitly):
            return Selected.Explicitly
        elif code.startswith(self.selected):
            return Selected.Implicitly
        else:
            return Ignored.Implicitly

    def was_ignored(self, code: str) -> Selected | Ignored:
        """Determine if the code has been ignored by the user.

        :param code:
            The code for the check that has been run.
        :returns:
            Selected.Implicitly if the ignored list is empty,
            Ignored.Explicitly if the ignored list is not empty and a match was
            found,
            Selected.Implicitly if the ignored list is not empty but no match
            was found.
        """
        if code.startswith(self.ignored_explicitly):
            return Ignored.Explicitly
        elif code.startswith(self.ignored):
            return Ignored.Implicitly
        else:
            return Selected.Implicitly

    def make_decision(self, code: str) -> Decision:
        """Decide if code should be ignored or selected."""
        selected = self.was_selected(code)
        ignored = self.was_ignored(code)
        LOG.debug('The user configured %r to be %r, %r', code, selected, ignored)
        if isinstance(selected, Selected) and isinstance(ignored, Selected):
            return Decision.Selected
        elif isinstance(selected, Ignored) and isinstance(ignored, Ignored):
            return Decision.Ignored
        elif selected is Selected.Explicitly and ignored is not Ignored.Explicitly:
            return Decision.Selected
        elif selected is not Selected.Explicitly and ignored is Ignored.Explicitly:
            return Decision.Ignored
        elif selected is Ignored.Implicitly and ignored is Selected.Implicitly:
            return Decision.Ignored
        elif selected is Selected.Explicitly and ignored is Ignored.Explicitly or (selected is Selected.Implicitly and ignored is Ignored.Implicitly):
            select = next((s for s in self.selected if code.startswith(s)))
            ignore = next((s for s in self.ignored if code.startswith(s)))
            if len(select) > len(ignore):
                return Decision.Selected
            else:
                return Decision.Ignored
        else:
            raise AssertionError(f'unreachable {code} {selected} {ignored}')

    def decision_for(self, code: str) -> Decision:
        """Return the decision for a specific code.

        This method caches the decisions for codes to avoid retracing the same
        logic over and over again. We only care about the select and ignore
        rules as specified by the user in their configuration files and
        command-line flags.

        This method does not look at whether the specific line is being
        ignored in the file itself.

        :param code: The code for the check that has been run.
        """
        decision = self.cache.get(code)
        if decision is None:
            decision = self.make_decision(code)
            self.cache[code] = decision
            LOG.debug('"%s" will be "%s"', code, decision)
        return decision