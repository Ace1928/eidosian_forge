import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union
class DefaultFilter(BaseFilter):
    """
    The default filter, which ignores files and directories that you might commonly want to ignore.
    """
    ignore_dirs: Sequence[str] = ('__pycache__', '.git', '.hg', '.svn', '.tox', '.venv', 'site-packages', '.idea', 'node_modules', '.mypy_cache', '.pytest_cache', '.hypothesis')
    'Directory names to ignore.'
    ignore_entity_patterns: Sequence[str] = ('\\.py[cod]$', '\\.___jb_...___$', '\\.sw.$', '~$', '^\\.\\#', '^\\.DS_Store$', '^flycheck_')
    'File/Directory name patterns to ignore.'

    def __init__(self, *, ignore_dirs: Optional[Sequence[str]]=None, ignore_entity_patterns: Optional[Sequence[str]]=None, ignore_paths: Optional[Sequence[Union[str, Path]]]=None) -> None:
        """
        Args:
            ignore_dirs: if not `None`, overrides the `ignore_dirs` value set on the class.
            ignore_entity_patterns: if not `None`, overrides the `ignore_entity_patterns` value set on the class.
            ignore_paths: if not `None`, overrides the `ignore_paths` value set on the class.
        """
        if ignore_dirs is not None:
            self.ignore_dirs = ignore_dirs
        if ignore_entity_patterns is not None:
            self.ignore_entity_patterns = ignore_entity_patterns
        if ignore_paths is not None:
            self.ignore_paths = ignore_paths
        super().__init__()