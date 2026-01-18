import logging
import os
import sys
import sysconfig
import typing
from pip._internal.exceptions import InvalidSchemeCombination, UserInstallationInvalid
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import change_root, get_major_minor_version, is_osx_framework
def _should_use_osx_framework_prefix() -> bool:
    """Check for Apple's ``osx_framework_library`` scheme.

    Python distributed by Apple's Command Line Tools has this special scheme
    that's used when:

    * This is a framework build.
    * We are installing into the system prefix.

    This does not account for ``pip install --prefix`` (also means we're not
    installing to the system prefix), which should use ``posix_prefix``, but
    logic here means ``_infer_prefix()`` outputs ``osx_framework_library``. But
    since ``prefix`` is not available for ``sysconfig.get_default_scheme()``,
    which is the stdlib replacement for ``_infer_prefix()``, presumably Apple
    wouldn't be able to magically switch between ``osx_framework_library`` and
    ``posix_prefix``. ``_infer_prefix()`` returning ``osx_framework_library``
    means its behavior is consistent whether we use the stdlib implementation
    or our own, and we deal with this special case in ``get_scheme()`` instead.
    """
    return 'osx_framework_library' in _AVAILABLE_SCHEMES and (not running_under_virtualenv()) and is_osx_framework()