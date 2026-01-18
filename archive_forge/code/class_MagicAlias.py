import os
import re
import sys
from getopt import getopt, GetoptError
from traitlets.config.configurable import Configurable
from . import oinspect
from .error import UsageError
from .inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from ..utils.ipstruct import Struct
from ..utils.process import arg_split
from ..utils.text import dedent
from traitlets import Bool, Dict, Instance, observe
from logging import error
import typing as t
class MagicAlias(object):
    """An alias to another magic function.

    An alias is determined by its magic name and magic kind. Lookup
    is done at call time, so if the underlying magic changes the alias
    will call the new function.

    Use the :meth:`MagicsManager.register_alias` method or the
    `%alias_magic` magic function to create and register a new alias.
    """

    def __init__(self, shell, magic_name, magic_kind, magic_params=None):
        self.shell = shell
        self.magic_name = magic_name
        self.magic_params = magic_params
        self.magic_kind = magic_kind
        self.pretty_target = '%s%s' % (magic_escapes[self.magic_kind], self.magic_name)
        self.__doc__ = 'Alias for `%s`.' % self.pretty_target
        self._in_call = False

    def __call__(self, *args, **kwargs):
        """Call the magic alias."""
        fn = self.shell.find_magic(self.magic_name, self.magic_kind)
        if fn is None:
            raise UsageError('Magic `%s` not found.' % self.pretty_target)
        if self._in_call:
            raise UsageError('Infinite recursion detected; magic aliases cannot call themselves.')
        self._in_call = True
        try:
            if self.magic_params:
                args_list = list(args)
                args_list[0] = self.magic_params + ' ' + args[0]
                args = tuple(args_list)
            return fn(*args, **kwargs)
        finally:
            self._in_call = False