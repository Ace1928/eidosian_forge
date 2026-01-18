import argparse
import collections
import logging
import sys
import curtsies
import curtsies.events
import curtsies.input
import curtsies.window
from . import args as bpargs, translations, inspection
from .config import Config
from .curtsiesfrontend import events
from .curtsiesfrontend.coderunner import SystemExitFromCodeRunner
from .curtsiesfrontend.interpreter import Interp
from .curtsiesfrontend.repl import BaseRepl
from .repl import extract_exit_value
from .translations import _
from typing import (
from ._typing_compat import Protocol
class FullCurtsiesRepl(BaseRepl):

    def __init__(self, config: Config, locals_: Optional[Dict[str, Any]]=None, banner: Optional[str]=None, interp: Optional[Interp]=None) -> None:
        self.input_generator = curtsies.input.Input(keynames='curtsies', sigint_event=True, paste_threshold=None)
        window = curtsies.window.CursorAwareWindow(sys.stdout, sys.stdin, keep_last_line=True, hide_cursor=False, extra_bytes_callback=self.input_generator.unget_bytes)
        self._request_refresh_callback: Callable[[], None] = self.input_generator.event_trigger(events.RefreshRequestEvent)
        self._schedule_refresh_callback = self.input_generator.scheduled_event_trigger(events.ScheduledRefreshRequestEvent)
        self._request_reload_callback = self.input_generator.threadsafe_event_trigger(events.ReloadEvent)
        self._interrupting_refresh_callback = self.input_generator.threadsafe_event_trigger(lambda: None)
        self._request_undo_callback = self.input_generator.event_trigger(events.UndoEvent)
        with self.input_generator:
            pass
        super().__init__(config, window, locals_=locals_, banner=banner, interp=interp, orig_tcattrs=self.input_generator.original_stty)

    def _request_refresh(self) -> None:
        return self._request_refresh_callback()

    def _schedule_refresh(self, when: float) -> None:
        return self._schedule_refresh_callback(when)

    def _request_reload(self, files_modified: Sequence[str]) -> None:
        return self._request_reload_callback(files_modified=files_modified)

    def interrupting_refresh(self) -> None:
        return self._interrupting_refresh_callback()

    def request_undo(self, n: int=1) -> None:
        return self._request_undo_callback(n=n)

    def get_term_hw(self) -> Tuple[int, int]:
        return self.window.get_term_hw()

    def get_cursor_vertical_diff(self) -> int:
        return self.window.get_cursor_vertical_diff()

    def get_top_usable_line(self) -> int:
        return self.window.top_usable_row

    def on_suspend(self) -> None:
        self.window.__exit__(None, None, None)
        self.input_generator.__exit__(None, None, None)

    def after_suspend(self) -> None:
        self.input_generator.__enter__()
        self.window.__enter__()
        self.interrupting_refresh()

    def process_event_and_paint(self, e: Union[str, curtsies.events.Event, None]) -> None:
        """If None is passed in, just paint the screen"""
        try:
            if e is not None:
                self.process_event(e)
        except (SystemExitFromCodeRunner, SystemExit) as err:
            array, cursor_pos = self.paint(about_to_exit=True, user_quit=isinstance(err, SystemExitFromCodeRunner))
            scrolled = self.window.render_to_terminal(array, cursor_pos)
            self.scroll_offset += scrolled
            raise
        else:
            array, cursor_pos = self.paint()
            scrolled = self.window.render_to_terminal(array, cursor_pos)
            self.scroll_offset += scrolled

    def mainloop(self, interactive: bool=True, paste: Optional[curtsies.events.PasteEvent]=None) -> None:
        if interactive:
            self.initialize_interp()
            self.process_event(events.RunStartupFileEvent())
        if paste:
            self.process_event(paste)
        self.process_event_and_paint(None)
        inputs = combined_events(self.input_generator)
        while self.module_gatherer.find_coroutine():
            e = inputs.send(0)
            if e is not None:
                self.process_event_and_paint(e)
        for e in inputs:
            self.process_event_and_paint(e)