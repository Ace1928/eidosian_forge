import re
from git.cmd import handle_process_output
from git.compat import defenc
from git.util import finalize_process, hex_to_bin
from .objects.blob import Blob
from .objects.util import mode_str_to_int
from typing import (
from git.types import PathLike, Literal
@staticmethod
def _handle_diff_line(lines_bytes: bytes, repo: 'Repo', index: DiffIndex) -> None:
    lines = lines_bytes.decode(defenc)
    _, _, lines = lines.partition(':')
    for line in lines.split('\x00:'):
        if not line:
            continue
        meta, _, path = line.partition('\x00')
        path = path.rstrip('\x00')
        a_blob_id: Optional[str]
        b_blob_id: Optional[str]
        old_mode, new_mode, a_blob_id, b_blob_id, _change_type = meta.split(None, 4)
        change_type: Lit_change_type = cast(Lit_change_type, _change_type[0])
        score_str = ''.join(_change_type[1:])
        score = int(score_str) if score_str.isdigit() else None
        path = path.strip()
        a_path = path.encode(defenc)
        b_path = path.encode(defenc)
        deleted_file = False
        new_file = False
        copied_file = False
        rename_from = None
        rename_to = None
        if change_type == 'D':
            b_blob_id = None
            deleted_file = True
        elif change_type == 'A':
            a_blob_id = None
            new_file = True
        elif change_type == 'C':
            copied_file = True
            a_path_str, b_path_str = path.split('\x00', 1)
            a_path = a_path_str.encode(defenc)
            b_path = b_path_str.encode(defenc)
        elif change_type == 'R':
            a_path_str, b_path_str = path.split('\x00', 1)
            a_path = a_path_str.encode(defenc)
            b_path = b_path_str.encode(defenc)
            rename_from, rename_to = (a_path, b_path)
        elif change_type == 'T':
            pass
        diff = Diff(repo, a_path, b_path, a_blob_id, b_blob_id, old_mode, new_mode, new_file, deleted_file, copied_file, rename_from, rename_to, '', change_type, score)
        index.append(diff)