import gc
from io import BytesIO
import logging
import os
import os.path as osp
import stat
import uuid
import git
from git.cmd import Git
from git.compat import defenc
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import (
from git.objects.base import IndexObject, Object
from git.objects.util import TraversableIterableObj
from git.util import (
from .util import (
from typing import Callable, Dict, Mapping, Sequence, TYPE_CHECKING, cast
from typing import Any, Iterator, Union
from git.types import Commit_ish, Literal, PathLike, TBD
@classmethod
def _write_git_file_and_module_config(cls, working_tree_dir: PathLike, module_abspath: PathLike) -> None:
    """Write a .git file containing a(preferably) relative path to the actual git module repository.

        It is an error if the module_abspath cannot be made into a relative path, relative to the working_tree_dir

        :note: This will overwrite existing files!
        :note: as we rewrite both the git file as well as the module configuration, we might fail on the configuration
            and will not roll back changes done to the git file. This should be a non - issue, but may easily be fixed
            if it becomes one.
        :param working_tree_dir: Directory to write the .git file into
        :param module_abspath: Absolute path to the bare repository
        """
    git_file = osp.join(working_tree_dir, '.git')
    rela_path = osp.relpath(module_abspath, start=working_tree_dir)
    if os.name == 'nt' and osp.isfile(git_file):
        os.remove(git_file)
    with open(git_file, 'wb') as fp:
        fp.write(('gitdir: %s' % rela_path).encode(defenc))
    with GitConfigParser(osp.join(module_abspath, 'config'), read_only=False, merge_includes=False) as writer:
        writer.set_value('core', 'worktree', to_native_path_linux(osp.relpath(working_tree_dir, start=module_abspath)))