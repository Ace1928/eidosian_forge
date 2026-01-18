import os
from argparse import Namespace, _SubParsersAction
from functools import wraps
from tempfile import mkstemp
from typing import Any, Callable, Iterable, List, Optional, Union
from ..utils import CachedRepoInfo, CachedRevisionInfo, HFCacheInfo, scan_cache_dir
from . import BaseHuggingfaceCLICommand
from ._cli_utils import ANSI
def _manual_review_no_tui(hf_cache_info: HFCacheInfo, preselected: List[str]) -> List[str]:
    """Ask the user for a manual review of the revisions to delete.

    Used when TUI is disabled. Manual review happens in a separate tmp file that the
    user can manually edit.
    """
    fd, tmp_path = mkstemp(suffix='.txt')
    os.close(fd)
    lines = []
    for repo in sorted(hf_cache_info.repos, key=_repo_sorting_order):
        lines.append(f'\n# {repo.repo_type.capitalize()} {repo.repo_id} ({repo.size_on_disk_str}, used {repo.last_accessed_str})')
        for revision in sorted(repo.revisions, key=_revision_sorting_order):
            lines.append(f'{('' if revision.commit_hash in preselected else '#')}    {revision.commit_hash} # Refs: {', '.join(sorted(revision.refs)) or '(detached)'} # modified {revision.last_modified_str}')
    with open(tmp_path, 'w') as f:
        f.write(_MANUAL_REVIEW_NO_TUI_INSTRUCTIONS)
        f.write('\n'.join(lines))
    instructions = f'\n    TUI is disabled. In order to select which revisions you want to delete, please edit\n    the following file using the text editor of your choice. Instructions for manual\n    editing are located at the beginning of the file. Edit the file, save it and confirm\n    to continue.\n    File to edit: {ANSI.bold(tmp_path)}\n    '
    print('\n'.join((line.strip() for line in instructions.strip().split('\n'))))
    while True:
        selected_hashes = _read_manual_review_tmp_file(tmp_path)
        if _ask_for_confirmation_no_tui(_get_expectations_str(hf_cache_info, selected_hashes) + ' Continue ?', default=False):
            break
    os.remove(tmp_path)
    return selected_hashes