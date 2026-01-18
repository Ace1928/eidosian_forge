from __future__ import (absolute_import, division, print_function)
from abc import ABCMeta
from collections.abc import Container, Mapping, Sequence, Set
from ansible.module_utils.common.collections import ImmutableDict
from ansible.module_utils.six import add_metaclass, binary_type, text_type
from ansible.utils.singleton import Singleton
class CLIArgs(ImmutableDict):
    """
    Hold a parsed copy of cli arguments

    We have both this non-Singleton version and the Singleton, GlobalCLIArgs, version to leave us
    room to implement a Context object in the future.  Whereas there should only be one set of args
    in a global context, individual Context objects might want to pretend that they have different
    command line switches to trigger different behaviour when they run.  So if we support Contexts
    in the future, they would use CLIArgs instead of GlobalCLIArgs to store their version of command
    line flags.
    """

    def __init__(self, mapping):
        toplevel = {}
        for key, value in mapping.items():
            toplevel[key] = _make_immutable(value)
        super(CLIArgs, self).__init__(toplevel)

    @classmethod
    def from_options(cls, options):
        return cls(vars(options))