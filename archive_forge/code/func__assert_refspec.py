import contextlib
import logging
import re
from git.cmd import Git, handle_process_output
from git.compat import defenc, force_text
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import GitCommandError
from git.refs import Head, Reference, RemoteReference, SymbolicReference, TagReference
from git.util import (
from typing import (
from git.types import PathLike, Literal, Commit_ish
def _assert_refspec(self) -> None:
    """Turns out we can't deal with remotes if the refspec is missing."""
    config = self.config_reader
    unset = 'placeholder'
    try:
        if config.get_value('fetch', default=unset) is unset:
            msg = "Remote '%s' has no refspec set.\n"
            msg += 'You can set it as follows:'
            msg += ' \'git config --add "remote.%s.fetch +refs/heads/*:refs/heads/*"\'.'
            raise AssertionError(msg % (self.name, self.name))
    finally:
        config.release()