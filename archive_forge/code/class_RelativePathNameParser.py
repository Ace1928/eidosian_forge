from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
class RelativePathNameParser(DynamicChoicesParser):
    """Composite argument parser for relative path names."""
    RELATIVE_NAMES = ['.', '..']

    def __init__(self, choices: list[str]) -> None:
        self.choices = choices
        super().__init__()

    def get_choices(self, value: str) -> list[str]:
        """Return a list of valid choices based on the given input value."""
        choices = list(self.choices)
        if value in self.RELATIVE_NAMES:
            choices.extend((f'{item}{PATH_DELIMITER}' for item in self.RELATIVE_NAMES))
        return choices