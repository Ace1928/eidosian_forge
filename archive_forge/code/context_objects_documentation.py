from __future__ import (absolute_import, division, print_function)
from abc import ABCMeta
from collections.abc import Container, Mapping, Sequence, Set
from ansible.module_utils.common.collections import ImmutableDict
from ansible.module_utils.six import add_metaclass, binary_type, text_type
from ansible.utils.singleton import Singleton

    Globally hold a parsed copy of cli arguments.

    Only one of these exist per program as it is for global context
    