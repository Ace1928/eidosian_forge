import sys
from optparse import Values
from typing import AbstractSet, List
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.operations.freeze import freeze
from pip._internal.utils.compat import stdlib_pkgs
def _dev_pkgs() -> AbstractSet[str]:
    pkgs = {'pip'}
    if _should_suppress_build_backends():
        pkgs |= {'setuptools', 'distribute', 'wheel'}
    return pkgs