import fnmatch
import getpass
import os
import re
import shlex
import socket
from hashlib import sha1
from io import StringIO
from functools import partial
from .ssh_exception import CouldNotCanonicalize, ConfigParseError
def _allowed_tokens(self, key):
    """
        Given config ``key``, return list of token strings to tokenize.

        .. note::
            This feels like it wants to eventually go away, but is used to
            preserve as-strict-as-possible compatibility with OpenSSH, which
            for whatever reason only applies some tokens to some config keys.
        """
    return self.TOKENS_BY_CONFIG_KEY.get(key, [])