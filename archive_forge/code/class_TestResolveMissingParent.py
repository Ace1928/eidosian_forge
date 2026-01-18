import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
class TestResolveMissingParent(TestResolveConflicts):
    preamble = "\n$ brz init trunk\n...\n$ cd trunk\n$ mkdir dir\n$ echo 'trunk content' >dir/file\n$ brz add -q\n$ brz commit -m 'Create trunk' -q\n$ echo 'trunk content' >dir/file2\n$ brz add -q dir/file2\n$ brz commit -q -m 'Add dir/file2 in branch'\n$ brz branch -q . -r 1 ../branch\n$ cd ../branch\n$ brz rm -q dir/file --no-backup\n$ brz rm -q dir\n$ brz commit -q -m 'Remove dir/file'\n$ brz merge ../trunk\n2>+N  dir/\n2>+N  dir/file2\n2>Conflict adding files to dir.  Created directory.\n2>Conflict because dir is not versioned, but has versioned children.  Versioned directory.\n2>2 conflicts encountered.\n"

    def test_keep_them_all(self):
        self.run_script("\n$ brz resolve dir\n2>2 conflicts resolved, 0 remaining\n$ brz commit -q --strict -m 'No more conflicts nor unknown files'\n")

    def test_adopt_child(self):
        self.run_script("\n$ brz mv -q dir/file2 file2\n$ brz rm -q dir --no-backup\n$ brz resolve dir\n2>2 conflicts resolved, 0 remaining\n$ brz commit -q --strict -m 'No more conflicts nor unknown files'\n")

    def test_kill_them_all(self):
        self.run_script("\n$ brz rm -q dir --no-backup\n$ brz resolve dir\n2>2 conflicts resolved, 0 remaining\n$ brz commit -q --strict -m 'No more conflicts nor unknown files'\n")

    def test_resolve_taking_this(self):
        self.run_script("\n$ brz resolve --take-this dir\n2>...\n$ brz commit -q --strict -m 'No more conflicts nor unknown files'\n")

    def test_resolve_taking_other(self):
        self.run_script("\n$ brz resolve --take-other dir\n2>...\n$ brz commit -q --strict -m 'No more conflicts nor unknown files'\n")