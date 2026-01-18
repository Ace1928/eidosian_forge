import contextlib
import logging
import re
from git.cmd import Git, handle_process_output
from git.compat import defenc, force_text
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import GitCommandError
from git.refs import Head, Reference, RemoteReference, SymbolicReference, TagReference
from git.util import (
from typing import (
from git.types import PathLike, Literal, Commit_ish
def _get_fetch_info_from_stderr(self, proc: 'Git.AutoInterrupt', progress: Union[Callable[..., Any], RemoteProgress, None], kill_after_timeout: Union[None, float]=None) -> IterableList['FetchInfo']:
    progress = to_progress_instance(progress)
    output: IterableList['FetchInfo'] = IterableList('name')
    fetch_info_lines = []
    cmds = set(FetchInfo._flag_map.keys())
    progress_handler = progress.new_message_handler()
    handle_process_output(proc, None, progress_handler, finalizer=None, decode_streams=False, kill_after_timeout=kill_after_timeout)
    stderr_text = progress.error_lines and '\n'.join(progress.error_lines) or ''
    proc.wait(stderr=stderr_text)
    if stderr_text:
        _logger.warning('Error lines received while fetching: %s', stderr_text)
    for line in progress.other_lines:
        line = force_text(line)
        for cmd in cmds:
            if len(line) > 1 and line[0] == ' ' and (line[1] == cmd):
                fetch_info_lines.append(line)
                continue
    fetch_head = SymbolicReference(self.repo, 'FETCH_HEAD')
    with open(fetch_head.abspath, 'rb') as fp:
        fetch_head_info = [line.decode(defenc) for line in fp.readlines()]
    l_fil = len(fetch_info_lines)
    l_fhi = len(fetch_head_info)
    if l_fil != l_fhi:
        msg = 'Fetch head lines do not match lines provided via progress information\n'
        msg += 'length of progress lines %i should be equal to lines in FETCH_HEAD file %i\n'
        msg += 'Will ignore extra progress lines or fetch head lines.'
        msg %= (l_fil, l_fhi)
        _logger.debug(msg)
        _logger.debug(b'info lines: ' + str(fetch_info_lines).encode('UTF-8'))
        _logger.debug(b'head info: ' + str(fetch_head_info).encode('UTF-8'))
        if l_fil < l_fhi:
            fetch_head_info = fetch_head_info[:l_fil]
        else:
            fetch_info_lines = fetch_info_lines[:l_fhi]
    for err_line, fetch_line in zip(fetch_info_lines, fetch_head_info):
        try:
            output.append(FetchInfo._from_line(self.repo, err_line, fetch_line))
        except ValueError as exc:
            _logger.debug('Caught error while parsing line: %s', exc)
            _logger.warning('Git informed while fetching: %s', err_line.strip())
    return output