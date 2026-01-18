import contextlib
import errno
import os
import time
from stat import S_IEXEC, S_ISREG
from typing import Callable
from . import config as _mod_config
from . import controldir, errors, lazy_import, lock, osutils, registry, trace
from breezy import (
from breezy.i18n import gettext
from .errors import BzrError, DuplicateKey, InternalBzrError
from .filters import ContentFilterContext, filtered_output_bytes
from .mutabletree import MutableTree
from .osutils import delete_any, file_kind, pathjoin, sha_file, splitpath
from .progress import ProgressPhase
from .transport import FileExists, NoSuchFile
from .tree import InterTree, find_previous_path
def _alter_files(es, working_tree, target_tree, tt, pb, specific_files, backups, merge_modified, basis_tree=None):
    if basis_tree is not None:
        es.enter_context(basis_tree.lock_read())
    change_list = working_tree.iter_changes(target_tree, specific_files=specific_files, pb=pb)
    if not target_tree.is_versioned(''):
        skip_root = True
    else:
        skip_root = False
    deferred_files = []
    for id_num, change in enumerate(change_list):
        target_path, wt_path = change.path
        target_versioned, wt_versioned = change.versioned
        target_name, wt_name = change.name
        target_kind, wt_kind = change.kind
        target_executable, wt_executable = change.executable
        if skip_root and wt_path == '':
            continue
        mode_id = None
        if wt_path is not None:
            trans_id = tt.trans_id_tree_path(wt_path)
        else:
            trans_id = tt.assign_id()
        if change.changed_content:
            keep_content = False
            if wt_kind == 'file' and (backups or target_kind is None):
                wt_sha1 = working_tree.get_file_sha1(wt_path)
                if merge_modified.get(wt_path) != wt_sha1:
                    if basis_tree is None:
                        basis_tree = working_tree.basis_tree()
                        es.enter_context(basis_tree.lock_read())
                    basis_inter = InterTree.get(basis_tree, working_tree)
                    basis_path = basis_inter.find_source_path(wt_path)
                    if basis_path is None:
                        if target_kind is None and (not target_versioned):
                            keep_content = True
                    elif wt_sha1 != basis_tree.get_file_sha1(basis_path):
                        keep_content = True
            if wt_kind is not None:
                if not keep_content:
                    tt.delete_contents(trans_id)
                elif target_kind is not None:
                    parent_trans_id = tt.trans_id_tree_path(osutils.dirname(wt_path))
                    backup_name = tt._available_backup_name(wt_name, parent_trans_id)
                    tt.adjust_path(backup_name, parent_trans_id, trans_id)
                    new_trans_id = tt.create_path(wt_name, parent_trans_id)
                    if wt_versioned and target_versioned:
                        tt.unversion_file(trans_id)
                        tt.version_file(new_trans_id, file_id=getattr(change, 'file_id', None))
                    mode_id = trans_id
                    trans_id = new_trans_id
            if target_kind in ('directory', 'tree-reference'):
                tt.create_directory(trans_id)
                if target_kind == 'tree-reference':
                    revision = target_tree.get_reference_revision(target_path)
                    tt.set_tree_reference(revision, trans_id)
            elif target_kind == 'symlink':
                tt.create_symlink(target_tree.get_symlink_target(target_path), trans_id)
            elif target_kind == 'file':
                deferred_files.append((target_path, (trans_id, mode_id, target_path)))
                if basis_tree is None:
                    basis_tree = working_tree.basis_tree()
                    es.enter_context(basis_tree.lock_read())
                new_sha1 = target_tree.get_file_sha1(target_path)
                basis_inter = InterTree.get(basis_tree, target_tree)
                basis_path = basis_inter.find_source_path(target_path)
                if basis_path is not None and new_sha1 == basis_tree.get_file_sha1(basis_path):
                    if basis_path in merge_modified:
                        del merge_modified[basis_path]
                else:
                    merge_modified[target_path] = new_sha1
                if keep_content and wt_executable == target_executable:
                    tt.set_executability(target_executable, trans_id)
            elif target_kind is not None:
                raise AssertionError(target_kind)
        if not wt_versioned and target_versioned:
            tt.version_file(trans_id, file_id=getattr(change, 'file_id', None))
        if wt_versioned and (not target_versioned):
            tt.unversion_file(trans_id)
        if target_name is not None and (wt_name != target_name or change.is_reparented()):
            if target_path == '':
                parent_trans = ROOT_PARENT
            else:
                target_parent = change.parent_id[0]
                parent_trans = tt.trans_id_file_id(target_parent)
            if wt_path == '' and wt_versioned:
                tt.adjust_root_path(target_name, parent_trans)
            else:
                tt.adjust_path(target_name, parent_trans, trans_id)
        if wt_executable != target_executable and target_kind == 'file':
            tt.set_executability(target_executable, trans_id)
    if working_tree.supports_content_filtering():
        for (trans_id, mode_id, target_path), bytes in target_tree.iter_files_bytes(deferred_files):
            filters = working_tree._content_filter_stack(target_path)
            bytes = filtered_output_bytes(bytes, filters, ContentFilterContext(target_path, working_tree))
            tt.create_file(bytes, trans_id, mode_id)
    else:
        for (trans_id, mode_id, target_path), bytes in target_tree.iter_files_bytes(deferred_files):
            tt.create_file(bytes, trans_id, mode_id)
    tt.fixup_new_roots()
    return merge_modified