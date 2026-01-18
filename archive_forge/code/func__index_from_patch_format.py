import re
from git.cmd import handle_process_output
from git.compat import defenc
from git.util import finalize_process, hex_to_bin
from .objects.blob import Blob
from .objects.util import mode_str_to_int
from typing import (
from git.types import PathLike, Literal
@classmethod
def _index_from_patch_format(cls, repo: 'Repo', proc: Union['Popen', 'Git.AutoInterrupt']) -> DiffIndex:
    """Create a new DiffIndex from the given process output which must be in patch format.

        :param repo: The repository we are operating on
        :param proc: ``git diff`` process to read from (supports :class:`Git.AutoInterrupt` wrapper)
        :return: git.DiffIndex
        """
    text_list: List[bytes] = []
    handle_process_output(proc, text_list.append, None, finalize_process, decode_streams=False)
    text = b''.join(text_list)
    index: 'DiffIndex' = DiffIndex()
    previous_header: Union[Match[bytes], None] = None
    header: Union[Match[bytes], None] = None
    a_path, b_path = (None, None)
    a_mode, b_mode = (None, None)
    for _header in cls.re_header.finditer(text):
        a_path_fallback, b_path_fallback, old_mode, new_mode, rename_from, rename_to, new_file_mode, deleted_file_mode, copied_file_name, a_blob_id, b_blob_id, b_mode, a_path, b_path = _header.groups()
        new_file, deleted_file, copied_file = (bool(new_file_mode), bool(deleted_file_mode), bool(copied_file_name))
        a_path = cls._pick_best_path(a_path, rename_from, a_path_fallback)
        b_path = cls._pick_best_path(b_path, rename_to, b_path_fallback)
        if previous_header is not None:
            index[-1].diff = text[previous_header.end():_header.start()]
        a_mode = old_mode or deleted_file_mode or (a_path and (b_mode or new_mode or new_file_mode))
        b_mode = b_mode or new_mode or new_file_mode or (b_path and a_mode)
        index.append(Diff(repo, a_path, b_path, a_blob_id and a_blob_id.decode(defenc), b_blob_id and b_blob_id.decode(defenc), a_mode and a_mode.decode(defenc), b_mode and b_mode.decode(defenc), new_file, deleted_file, copied_file, rename_from, rename_to, None, None, None))
        previous_header = _header
        header = _header
    if index and header:
        index[-1].diff = text[header.end():]
    return index