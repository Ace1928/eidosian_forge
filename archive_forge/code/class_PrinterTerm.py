import itertools
import platform
import sys
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union
import click
import wandb
from . import ipython, sparkline
class PrinterTerm(_Printer):

    def __init__(self) -> None:
        super().__init__()
        self._html = False
        self._progress = itertools.cycle(['-', '\\', '|', '/'])

    def _display(self, text: Union[str, List[str], Tuple[str]], *, level: Optional[Union[str, int]]=None, default_text: Optional[Union[str, List[str], Tuple[str]]]=None) -> None:
        text = '\n'.join(text) if isinstance(text, (list, tuple)) else text
        if default_text is not None:
            default_text = '\n'.join(default_text) if isinstance(default_text, (list, tuple)) else default_text
            text = text or default_text
        self._display_fn_mapping(level)(text)

    @staticmethod
    def _display_fn_mapping(level: Optional[Union[str, int]]) -> Callable[[str], None]:
        level = _Printer._sanitize_level(level)
        if level >= CRITICAL:
            return wandb.termerror
        elif ERROR <= level < CRITICAL:
            return wandb.termerror
        elif WARNING <= level < ERROR:
            return wandb.termwarn
        elif INFO <= level < WARNING:
            return wandb.termlog
        elif DEBUG <= level < INFO:
            return wandb.termlog
        else:
            return wandb.termlog

    def progress_update(self, text: str, percent_done: Optional[float]=None) -> None:
        wandb.termlog(f'{next(self._progress)} {text}', newline=False)

    def progress_close(self, text: Optional[str]=None) -> None:
        text = text or ' ' * 79
        wandb.termlog(text)

    def code(self, text: str) -> str:
        ret: str = click.style(text, bold=True)
        return ret

    def name(self, text: str) -> str:
        ret: str = click.style(text, fg='yellow')
        return ret

    def link(self, link: str, text: Optional[str]=None) -> str:
        ret: str = click.style(link, fg='blue', underline=True)
        return ret

    def emoji(self, name: str) -> str:
        emojis = dict()
        if platform.system() != 'Windows' and wandb.util.is_unicode_safe(sys.stdout):
            emojis = dict(star='⭐️', broom='🧹', rocket='🚀', gorilla='🦍', turtle='🐢', lightning='️⚡')
        return emojis.get(name, '')

    def status(self, text: str, failure: Optional[bool]=None) -> str:
        color = 'red' if failure else 'green'
        ret: str = click.style(text, fg=color)
        return ret

    def files(self, text: str) -> str:
        ret: str = click.style(text, fg='magenta', bold=True)
        return ret

    def grid(self, rows: List[List[str]], title: Optional[str]=None) -> str:
        max_len = max((len(row[0]) for row in rows))
        format_row = ' '.join(['{:>{max_len}}', '{}' * (len(rows[0]) - 1)])
        grid = '\n'.join([format_row.format(*row, max_len=max_len) for row in rows])
        if title:
            return f'{title}\n{grid}\n'
        return f'{grid}\n'

    def panel(self, columns: List[str]) -> str:
        return '\n' + '\n'.join(columns)