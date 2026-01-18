import asyncio
import itertools
import logging
import subprocess
import textwrap
import types
from typing import List
class SubprocessCalledProcessError(subprocess.CalledProcessError):
    """The subprocess.CalledProcessError with stripped stdout."""
    LAST_N_LINES = 50

    def __init__(self, *args, cmd_index=None, **kwargs):
        self.cmd_index = cmd_index
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_last_n_line(str_data: str, last_n_lines: int) -> str:
        if last_n_lines < 0:
            return str_data
        lines = str_data.strip().split('\n')
        return '\n'.join(lines[-last_n_lines:])

    def __str__(self):
        str_list = [] if self.cmd_index is None else [f'Run cmd[{self.cmd_index}] failed with the following details.']
        str_list.append(super().__str__())
        out = {'stdout': self.stdout, 'stderr': self.stderr}
        for name, s in out.items():
            if s:
                subtitle = f'Last {self.LAST_N_LINES} lines of {name}:'
                last_n_line_str = self._get_last_n_line(s, self.LAST_N_LINES).strip()
                str_list.append(f'{subtitle}\n{textwrap.indent(last_n_line_str, ' ' * 4)}')
        return '\n'.join(str_list)