from __future__ import annotations
import os
import pathlib
from types import FrameType, ModuleType
from typing import (
class TConfigurable(Protocol):
    """Something that can proxy to the coverage configuration settings."""

    def get_option(self, option_name: str) -> TConfigValueOut | None:
        """Get an option from the configuration.

        `option_name` is a colon-separated string indicating the section and
        option name.  For example, the ``branch`` option in the ``[run]``
        section of the config file would be indicated with `"run:branch"`.

        Returns the value of the option.

        """

    def set_option(self, option_name: str, value: TConfigValueIn | TConfigSectionIn) -> None:
        """Set an option in the configuration.

        `option_name` is a colon-separated string indicating the section and
        option name.  For example, the ``branch`` option in the ``[run]``
        section of the config file would be indicated with `"run:branch"`.

        `value` is the new value for the option.

        """