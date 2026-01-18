import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
@dataclass
class TableSettings:
    vertical_strategy: str = 'lines'
    horizontal_strategy: str = 'lines'
    explicit_vertical_lines: list = None
    explicit_horizontal_lines: list = None
    snap_tolerance: float = DEFAULT_SNAP_TOLERANCE
    snap_x_tolerance: float = UNSET
    snap_y_tolerance: float = UNSET
    join_tolerance: float = DEFAULT_JOIN_TOLERANCE
    join_x_tolerance: float = UNSET
    join_y_tolerance: float = UNSET
    edge_min_length: float = 3
    min_words_vertical: float = DEFAULT_MIN_WORDS_VERTICAL
    min_words_horizontal: float = DEFAULT_MIN_WORDS_HORIZONTAL
    intersection_tolerance: float = 3
    intersection_x_tolerance: float = UNSET
    intersection_y_tolerance: float = UNSET
    text_settings: dict = None

    def __post_init__(self) -> 'TableSettings':
        """Clean up user-provided table settings.

        Validates that the table settings provided consists of acceptable values and
        returns a cleaned up version. The cleaned up version fills out the missing
        values with the default values in the provided settings.

        TODO: Can be further used to validate that the values are of the correct
            type. For example, raising a value error when a non-boolean input is
            provided for the key ``keep_blank_chars``.

        :param table_settings: User-provided table settings.
        :returns: A cleaned up version of the user-provided table settings.
        :raises ValueError: When an unrecognised key is provided.
        """
        for setting in NON_NEGATIVE_SETTINGS:
            if (getattr(self, setting) or 0) < 0:
                raise ValueError(f"Table setting '{setting}' cannot be negative")
        for orientation in ['horizontal', 'vertical']:
            strategy = getattr(self, orientation + '_strategy')
            if strategy not in TABLE_STRATEGIES:
                raise ValueError(f'{orientation}_strategy must be one of{{{','.join(TABLE_STRATEGIES)}}}')
        if self.text_settings is None:
            self.text_settings = {}
        for attr in ['x_tolerance', 'y_tolerance']:
            if attr not in self.text_settings:
                self.text_settings[attr] = self.text_settings.get('tolerance', 3)
        if 'tolerance' in self.text_settings:
            del self.text_settings['tolerance']
        for attr, fallback in [('snap_x_tolerance', 'snap_tolerance'), ('snap_y_tolerance', 'snap_tolerance'), ('join_x_tolerance', 'join_tolerance'), ('join_y_tolerance', 'join_tolerance'), ('intersection_x_tolerance', 'intersection_tolerance'), ('intersection_y_tolerance', 'intersection_tolerance')]:
            if getattr(self, attr) is UNSET:
                setattr(self, attr, getattr(self, fallback))
        return self

    @classmethod
    def resolve(cls, settings=None):
        if settings is None:
            return cls()
        elif isinstance(settings, cls):
            return settings
        elif isinstance(settings, dict):
            core_settings = {}
            text_settings = {}
            for k, v in settings.items():
                if k[:5] == 'text_':
                    text_settings[k[5:]] = v
                else:
                    core_settings[k] = v
            core_settings['text_settings'] = text_settings
            return cls(**core_settings)
        else:
            raise ValueError(f'Cannot resolve settings: {settings}')