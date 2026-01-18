import os
import re
import sys
from traitlets.config.configurable import Configurable
from .error import UsageError
from traitlets import List, Instance
from logging import error
import typing as t
class AliasManager(Configurable):
    default_aliases: List = List(default_aliases()).tag(config=True)
    user_aliases: List = List(default_value=[]).tag(config=True)
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)

    def __init__(self, shell=None, **kwargs):
        super(AliasManager, self).__init__(shell=shell, **kwargs)
        if self.shell is not None:
            self.linemagics = self.shell.magics_manager.magics['line']
            self.init_aliases()

    def init_aliases(self):
        for name, cmd in self.default_aliases + self.user_aliases:
            if cmd.startswith('ls ') and self.shell is not None and (self.shell.colors == 'NoColor'):
                cmd = cmd.replace(' --color', '')
            self.soft_define_alias(name, cmd)

    @property
    def aliases(self):
        return [(n, func.cmd) for n, func in self.linemagics.items() if isinstance(func, Alias)]

    def soft_define_alias(self, name, cmd):
        """Define an alias, but don't raise on an AliasError."""
        try:
            self.define_alias(name, cmd)
        except AliasError as e:
            error('Invalid alias: %s' % e)

    def define_alias(self, name, cmd):
        """Define a new alias after validating it.

        This will raise an :exc:`AliasError` if there are validation
        problems.
        """
        caller = Alias(shell=self.shell, name=name, cmd=cmd)
        self.shell.magics_manager.register_function(caller, magic_kind='line', magic_name=name)

    def get_alias(self, name):
        """Return an alias, or None if no alias by that name exists."""
        aname = self.linemagics.get(name, None)
        return aname if isinstance(aname, Alias) else None

    def is_alias(self, name):
        """Return whether or not a given name has been defined as an alias"""
        return self.get_alias(name) is not None

    def undefine_alias(self, name):
        if self.is_alias(name):
            del self.linemagics[name]
        else:
            raise ValueError('%s is not an alias' % name)

    def clear_aliases(self):
        for name, _ in self.aliases:
            self.undefine_alias(name)

    def retrieve_alias(self, name):
        """Retrieve the command to which an alias expands."""
        caller = self.get_alias(name)
        if caller:
            return caller.cmd
        else:
            raise ValueError('%s is not an alias' % name)