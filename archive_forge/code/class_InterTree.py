from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union,
from . import errors, lock, osutils
from . import revision as _mod_revision
from . import trace
from .inter import InterObject
class InterTree(InterObject[Tree]):
    """This class represents operations taking place between two Trees.

    Its instances have methods like 'compare' and contain references to the
    source and target trees these operations are to be carried out on.

    Clients of breezy should not need to use InterTree directly, rather they
    should use the convenience methods on Tree such as 'Tree.compare()' which
    will pass through to InterTree as appropriate.
    """
    if TYPE_CHECKING:
        from .workingtree import WorkingTreeFormat
    _matching_from_tree_format: Optional['WorkingTreeFormat'] = None
    _matching_to_tree_format: Optional['WorkingTreeFormat'] = None
    _optimisers = []

    @classmethod
    def is_compatible(kls, source, target):
        return True

    @classmethod
    def get(cls, source: Tree, target: Tree) -> 'InterTree':
        return cast(InterTree, super().get(source, target))

    def compare(self, want_unchanged: bool=False, specific_files: Optional[List[str]]=None, extra_trees: Optional[List[Tree]]=None, require_versioned: bool=False, include_root: bool=False, want_unversioned: bool=False):
        """Return the changes from source to target.

        Returns: A TreeDelta.
        Args:
          specific_files: An optional list of file paths to restrict the
            comparison to. When mapping filenames to ids, all matches in all
            trees (including optional extra_trees) are used, and all children of
            matched directories are included.
          want_unchanged: An optional boolean requesting the inclusion of
            unchanged entries in the result.
          extra_trees: An optional list of additional trees to use when
            mapping the contents of specific_files (paths) to file_ids.
          require_versioned: An optional boolean (defaults to False). When
            supplied and True all the 'specific_files' must be versioned, or
            a PathsNotVersionedError will be thrown.
          want_unversioned: Scan for unversioned paths.
        """
        from . import delta
        trees = [self.source]
        if extra_trees is not None:
            trees = trees + extra_trees
        with self.lock_read():
            return delta._compare_trees(self.source, self.target, want_unchanged, specific_files, include_root, extra_trees=extra_trees, require_versioned=require_versioned, want_unversioned=want_unversioned)

    def iter_changes(self, include_unchanged: bool=False, specific_files: Optional[List[str]]=None, pb=None, extra_trees: List[Tree]=[], require_versioned: bool=True, want_unversioned: bool=False):
        """Generate an iterator of changes between trees.

        A TreeChange object is returned.

        Changed_content is True if the file's content has changed.  This
        includes changes to its kind, and to a symlink's target.

        versioned, parent, name, kind, executable are tuples of (from, to).
        If a file is missing in a tree, its kind is None.

        Iteration is done in parent-to-child order, relative to the target
        tree.

        There is no guarantee that all paths are in sorted order: the
        requirement to expand the search due to renames may result in children
        that should be found early being found late in the search, after
        lexically later results have been returned.

        Args:
          require_versioned: Raise errors.PathsNotVersionedError if a
            path in the specific_files list is not versioned in one of
            source, target or extra_trees.
          specific_files: An optional list of file paths to restrict the
            comparison to. When mapping filenames to ids, all matches in all
            trees (including optional extra_trees) are used, and all children
            of matched directories are included. The parents in the target tree
            of the specific files up to and including the root of the tree are
            always evaluated for changes too.
          want_unversioned: Should unversioned files be returned in the
            output. An unversioned file is defined as one with (False, False)
            for the versioned pair.
        """
        raise NotImplementedError(self.iter_changes)

    def file_content_matches(self, source_path: str, target_path: str, source_stat=None, target_stat=None):
        """Check if two files are the same in the source and target trees.

        This only checks that the contents of the files are the same,
        it does not touch anything else.

        Args:
          source_path: Path of the file in the source tree
          target_path: Path of the file in the target tree
          source_stat: Optional stat value of the file in the source tree
          target_stat: Optional stat value of the file in the target tree
        Returns: Boolean indicating whether the files have the same contents
        """
        with self.lock_read():
            source_verifier_kind, source_verifier_data = self.source.get_file_verifier(source_path, source_stat)
            target_verifier_kind, target_verifier_data = self.target.get_file_verifier(target_path, target_stat)
            if source_verifier_kind == target_verifier_kind:
                return source_verifier_data == target_verifier_data
            if source_verifier_kind != 'SHA1':
                source_sha1 = self.source.get_file_sha1(source_path, source_stat)
            else:
                source_sha1 = source_verifier_data
            if target_verifier_kind != 'SHA1':
                target_sha1 = self.target.get_file_sha1(target_path, target_stat)
            else:
                target_sha1 = target_verifier_data
            return source_sha1 == target_sha1

    def find_target_path(self, path: str, recurse: str='none') -> Optional[str]:
        """Find target tree path.

        Args:
          path: Path to search for (exists in source)
        Returns: path in target, or None if there is no equivalent path.
        Raises:
          NoSuchFile: If the path doesn't exist in source
        """
        raise NotImplementedError(self.find_target_path)

    def find_source_path(self, path: str, recurse: str='none') -> Optional[str]:
        """Find the source tree path.

        Args:
          path: Path to search for (exists in target)
        Returns: path in source, or None if there is no equivalent path.
        Raises:
          NoSuchFile: if the path doesn't exist in target
        """
        raise NotImplementedError(self.find_source_path)

    def find_target_paths(self, paths: List[str], recurse='none') -> Dict[str, Optional[str]]:
        """Find target tree paths.

        Args:
          paths: Iterable over paths in target to search for
        Returns: Dictionary mapping from source paths to paths in target , or
            None if there is no equivalent path.
        """
        ret = {}
        for path in paths:
            ret[path] = self.find_target_path(path, recurse=recurse)
        return ret

    def find_source_paths(self, paths: List[str], recurse: str='none') -> Dict[str, Optional[str]]:
        """Find source tree paths.

        Args:
          paths: Iterable over paths in target to search for
        Returns: Dictionary mapping from target paths to paths in source, or
            None if there is no equivalent path.
        """
        ret = {}
        for path in paths:
            ret[path] = self.find_source_path(path, recurse=recurse)
        return ret