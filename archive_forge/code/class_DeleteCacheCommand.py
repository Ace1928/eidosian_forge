import os
from argparse import Namespace, _SubParsersAction
from functools import wraps
from tempfile import mkstemp
from typing import Any, Callable, Iterable, List, Optional, Union
from ..utils import CachedRepoInfo, CachedRevisionInfo, HFCacheInfo, scan_cache_dir
from . import BaseHuggingfaceCLICommand
from ._cli_utils import ANSI
class DeleteCacheCommand(BaseHuggingfaceCLICommand):

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        delete_cache_parser = parser.add_parser('delete-cache', help='Delete revisions from the cache directory.')
        delete_cache_parser.add_argument('--dir', type=str, default=None, help='cache directory (optional). Default to the default HuggingFace cache.')
        delete_cache_parser.add_argument('--disable-tui', action='store_true', help="Disable Terminal User Interface (TUI) mode. Useful if your platform/terminal doesn't support the multiselect menu.")
        delete_cache_parser.set_defaults(func=DeleteCacheCommand)

    def __init__(self, args: Namespace) -> None:
        self.cache_dir: Optional[str] = args.dir
        self.disable_tui: bool = args.disable_tui

    def run(self):
        """Run `delete-cache` command with or without TUI."""
        hf_cache_info = scan_cache_dir(self.cache_dir)
        if self.disable_tui:
            selected_hashes = _manual_review_no_tui(hf_cache_info, preselected=[])
        else:
            selected_hashes = _manual_review_tui(hf_cache_info, preselected=[])
        if len(selected_hashes) > 0 and _CANCEL_DELETION_STR not in selected_hashes:
            confirm_message = _get_expectations_str(hf_cache_info, selected_hashes) + ' Confirm deletion ?'
            if self.disable_tui:
                confirmed = _ask_for_confirmation_no_tui(confirm_message)
            else:
                confirmed = _ask_for_confirmation_tui(confirm_message)
            if confirmed:
                strategy = hf_cache_info.delete_revisions(*selected_hashes)
                print('Start deletion.')
                strategy.execute()
                print(f'Done. Deleted {len(strategy.repos)} repo(s) and {len(strategy.snapshots)} revision(s) for a total of {strategy.expected_freed_size_str}.')
                return
        print('Deletion is cancelled. Do nothing.')