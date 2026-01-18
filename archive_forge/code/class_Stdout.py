from __future__ import annotations
import asyncio
import traceback
from asyncio import get_running_loop
from typing import Any, Callable, Coroutine, TextIO, cast
import asyncssh
from prompt_toolkit.application.current import AppSession, create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
class Stdout:

    def write(s, data: str) -> None:
        try:
            if self._chan is not None:
                self._chan.write(data.replace('\n', '\r\n'))
        except BrokenPipeError:
            pass

    def isatty(s) -> bool:
        return True

    def flush(s) -> None:
        pass

    @property
    def encoding(s) -> str:
        assert self._chan is not None
        return str(self._chan._orig_chan.get_encoding()[0])