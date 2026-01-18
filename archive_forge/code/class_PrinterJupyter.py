import itertools
import platform
import sys
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union
import click
import wandb
from . import ipython, sparkline
class PrinterJupyter(_Printer):

    def __init__(self) -> None:
        super().__init__()
        self._html = True
        self._progress = ipython.jupyter_progress_bar()

    def _display(self, text: Union[str, List[str], Tuple[str]], *, level: Optional[Union[str, int]]=None, default_text: Optional[Union[str, List[str], Tuple[str]]]=None) -> None:
        text = '<br/>'.join(text) if isinstance(text, (list, tuple)) else text
        if default_text is not None:
            default_text = '<br/>'.join(default_text) if isinstance(default_text, (list, tuple)) else default_text
            text = text or default_text
        self._display_fn_mapping(level)(text)

    @staticmethod
    def _display_fn_mapping(level: Optional[Union[str, int]]) -> Callable[[str], None]:
        level = _Printer._sanitize_level(level)
        if level >= CRITICAL:
            return ipython.display_html
        elif ERROR <= level < CRITICAL:
            return ipython.display_html
        elif WARNING <= level < ERROR:
            return ipython.display_html
        elif INFO <= level < WARNING:
            return ipython.display_html
        elif DEBUG <= level < INFO:
            return ipython.display_html
        else:
            return ipython.display_html

    def code(self, text: str) -> str:
        return f'<code>{text}<code>'

    def name(self, text: str) -> str:
        return f'<strong style="color:#cdcd00">{text}</strong>'

    def link(self, link: str, text: Optional[str]=None) -> str:
        return f'<a href={link!r} target="_blank">{text or link}</a>'

    def emoji(self, name: str) -> str:
        return ''

    def status(self, text: str, failure: Optional[bool]=None) -> str:
        color = 'red' if failure else 'green'
        return f'<strong style="color:{color}">{text}</strong>'

    def files(self, text: str) -> str:
        return f'<code>{text}</code>'

    def progress_update(self, text: str, percent_done: float) -> None:
        if self._progress:
            self._progress.update(percent_done, text)

    def progress_close(self, _: Optional[str]=None) -> None:
        if self._progress:
            self._progress.close()

    def grid(self, rows: List[List[str]], title: Optional[str]=None) -> str:
        format_row = ''.join(['<tr>', '<td>{}</td>' * len(rows[0]), '</tr>'])
        grid = ''.join([format_row.format(*row) for row in rows])
        grid = f'<table class="wandb">{grid}</table>'
        if title:
            return f'<h3>{title}</h3><br/>{grid}<br/>'
        return f'{grid}<br/>'

    def panel(self, columns: List[str]) -> str:
        row = ''.join([f'<div class="wandb-col">{col}</div>' for col in columns])
        return f'{ipython.TABLE_STYLES}<div class="wandb-row">{row}</div>'