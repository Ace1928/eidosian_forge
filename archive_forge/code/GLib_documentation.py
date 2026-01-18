import warnings
import sys
import socket
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..module import get_introspection_module
from .._gi import (variant_type_from_string, source_new,
from ..overrides import override, deprecated, deprecated_attr
from gi import PyGIDeprecationWarning, version_info
from gi import _option as option
from gi import _gi
from gi._error import GError
Return a list of the element signatures of the topmost signature tuple.

        If the signature is not a tuple, it returns one element with the entire
        signature. If the signature is an empty tuple, the result is [].

        This is useful for e. g. iterating over method parameters which are
        passed as a single Variant.
        