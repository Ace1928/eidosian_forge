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
def _get_version_from_git(pre_version=None):
    """Calculate a version string from git.

    If the revision is tagged, return that. Otherwise calculate a semantic
    version description of the tree.

    The number of revisions since the last tag is included in the dev counter
    in the version for untagged versions.

    :param pre_version: If supplied use this as the target version rather than
        inferring one from the last tag + commit messages.
    """
    git_dir = git._run_git_functions()
    if git_dir:
        try:
            tagged = git._run_git_command(['describe', '--exact-match'], git_dir, throw_on_error=True).replace('-', '.')
            target_version = version.SemanticVersion.from_pip_string(tagged)
        except Exception:
            if pre_version:
                target_version = version.SemanticVersion.from_pip_string(pre_version)
            else:
                target_version = None
        result = _get_version_from_git_target(git_dir, target_version)
        return result.release_string()
    try:
        return unicode()
    except NameError:
        return ''