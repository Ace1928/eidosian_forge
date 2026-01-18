from __future__ import annotations
import argparse
from flake8 import defaults
from flake8.options.manager import OptionManager
class JobsArgument:
    """Type callback for the --jobs argument."""

    def __init__(self, arg: str) -> None:
        """Parse and validate the --jobs argument.

        :param arg: The argument passed by argparse for validation
        """
        self.is_auto = False
        self.n_jobs = -1
        if arg == 'auto':
            self.is_auto = True
        elif arg.isdigit():
            self.n_jobs = int(arg)
        else:
            raise argparse.ArgumentTypeError(f"{arg!r} must be 'auto' or an integer.")

    def __repr__(self) -> str:
        """Representation for debugging."""
        return f'{type(self).__name__}({str(self)!r})'

    def __str__(self) -> str:
        """Format our JobsArgument class."""
        return 'auto' if self.is_auto else str(self.n_jobs)