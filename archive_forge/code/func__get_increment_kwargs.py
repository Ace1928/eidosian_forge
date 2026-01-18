from __future__ import unicode_literals
from distutils.command import install as du_install
from distutils import log
import email
import email.errors
import os
import re
import sys
import warnings
import pkg_resources
import setuptools
from setuptools.command import develop
from setuptools.command import easy_install
from setuptools.command import egg_info
from setuptools.command import install
from setuptools.command import install_scripts
from setuptools.command import sdist
from pbr import extra_files
from pbr import git
from pbr import options
import pbr.pbr_json
from pbr import testr_command
from pbr import version
import threading
from %(module_name)s import %(import_target)s
import sys
from %(module_name)s import %(import_target)s
def _get_increment_kwargs(git_dir, tag):
    """Calculate the sort of semver increment needed from git history.

    Every commit from HEAD to tag is consider for Sem-Ver metadata lines.
    See the pbr docs for their syntax.

    :return: a dict of kwargs for passing into SemanticVersion.increment.
    """
    result = {}
    if tag:
        version_spec = tag + '..HEAD'
    else:
        version_spec = 'HEAD'
    changelog = git._run_git_command(['log', '--pretty=%B', version_spec], git_dir)
    symbols = set()
    header = 'sem-ver:'
    for line in changelog.split('\n'):
        line = line.lower().strip()
        if not line.lower().strip().startswith(header):
            continue
        new_symbols = line[len(header):].strip().split(',')
        symbols.update([symbol.strip() for symbol in new_symbols])

    def _handle_symbol(symbol, symbols, impact):
        if symbol in symbols:
            result[impact] = True
            symbols.discard(symbol)
    _handle_symbol('bugfix', symbols, 'patch')
    _handle_symbol('feature', symbols, 'minor')
    _handle_symbol('deprecation', symbols, 'minor')
    _handle_symbol('api-break', symbols, 'major')
    for symbol in symbols:
        log.info('[pbr] Unknown Sem-Ver symbol %r' % symbol)
    result.pop('patch', None)
    return result