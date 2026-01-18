import logging
import os
import shutil
import sys
import urllib.parse
from typing import (
from pip._internal.cli.spinners import SpinnerInterface
from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import (
from pip._internal.utils.subprocess import (
from pip._internal.utils.urls import get_url_scheme
class RevOptions:
    """
    Encapsulates a VCS-specific revision to install, along with any VCS
    install options.

    Instances of this class should be treated as if immutable.
    """

    def __init__(self, vc_class: Type['VersionControl'], rev: Optional[str]=None, extra_args: Optional[CommandArgs]=None) -> None:
        """
        Args:
          vc_class: a VersionControl subclass.
          rev: the name of the revision to install.
          extra_args: a list of extra options.
        """
        if extra_args is None:
            extra_args = []
        self.extra_args = extra_args
        self.rev = rev
        self.vc_class = vc_class
        self.branch_name: Optional[str] = None

    def __repr__(self) -> str:
        return f'<RevOptions {self.vc_class.name}: rev={self.rev!r}>'

    @property
    def arg_rev(self) -> Optional[str]:
        if self.rev is None:
            return self.vc_class.default_arg_rev
        return self.rev

    def to_args(self) -> CommandArgs:
        """
        Return the VCS-specific command arguments.
        """
        args: CommandArgs = []
        rev = self.arg_rev
        if rev is not None:
            args += self.vc_class.get_base_rev_args(rev)
        args += self.extra_args
        return args

    def to_display(self) -> str:
        if not self.rev:
            return ''
        return f' (to revision {self.rev})'

    def make_new(self, rev: str) -> 'RevOptions':
        """
        Make a copy of the current instance, but with a new rev.

        Args:
          rev: the name of the revision for the new object.
        """
        return self.vc_class.make_rev_options(rev, extra_args=self.extra_args)