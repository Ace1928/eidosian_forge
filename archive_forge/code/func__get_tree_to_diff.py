import contextlib
import difflib
import os
import re
import sys
from typing import List, Optional, Type, Union
from .lazy_import import lazy_import
import errno
import patiencediff
import subprocess
from breezy import (
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext
from . import errors, osutils
from . import transport as _mod_transport
from .registry import Registry
from .trace import mutter, note, warning
from .tree import FileTimestampUnavailable, Tree
def _get_tree_to_diff(spec, tree=None, branch=None, basis_is_default=True):
    if branch is None and tree is not None:
        branch = tree.branch
    if spec is None or spec.spec is None:
        if basis_is_default:
            if tree is not None:
                return tree.basis_tree()
            else:
                return branch.basis_tree()
        else:
            return tree
    return spec.as_tree(branch)