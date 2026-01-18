from __future__ import annotations
import weakref
from asyncio import Task, sleep
from collections import deque
from typing import TYPE_CHECKING, Any, Generator
from prompt_toolkit.application.current import get_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters.app import vi_navigation_mode
from prompt_toolkit.keys import Keys
from prompt_toolkit.utils import Event
from .key_bindings import Binding, KeyBindingsBase
class KeyProcessor:
    """
    Statemachine that receives :class:`KeyPress` instances and according to the
    key bindings in the given :class:`KeyBindings`, calls the matching handlers.

    ::

        p = KeyProcessor(key_bindings)

        # Send keys into the processor.
        p.feed(KeyPress(Keys.ControlX, '\x18'))
        p.feed(KeyPress(Keys.ControlC, '\x03')

        # Process all the keys in the queue.
        p.process_keys()

        # Now the ControlX-ControlC callback will be called if this sequence is
        # registered in the key bindings.

    :param key_bindings: `KeyBindingsBase` instance.
    """

    def __init__(self, key_bindings: KeyBindingsBase) -> None:
        self._bindings = key_bindings
        self.before_key_press = Event(self)
        self.after_key_press = Event(self)
        self._flush_wait_task: Task[None] | None = None
        self.reset()

    def reset(self) -> None:
        self._previous_key_sequence: list[KeyPress] = []
        self._previous_handler: Binding | None = None
        self.input_queue: deque[KeyPress] = deque()
        self.key_buffer: list[KeyPress] = []
        self.arg: str | None = None
        self._process_coroutine = self._process()
        self._process_coroutine.send(None)

    def _get_matches(self, key_presses: list[KeyPress]) -> list[Binding]:
        """
        For a list of :class:`KeyPress` instances. Give the matching handlers
        that would handle this.
        """
        keys = tuple((k.key for k in key_presses))
        return [b for b in self._bindings.get_bindings_for_keys(keys) if b.filter()]

    def _is_prefix_of_longer_match(self, key_presses: list[KeyPress]) -> bool:
        """
        For a list of :class:`KeyPress` instances. Return True if there is any
        handler that is bound to a suffix of this keys.
        """
        keys = tuple((k.key for k in key_presses))
        filters = {b.filter for b in self._bindings.get_bindings_starting_with_keys(keys)}
        return any((f() for f in filters))

    def _process(self) -> Generator[None, KeyPress, None]:
        """
        Coroutine implementing the key match algorithm. Key strokes are sent
        into this generator, and it calls the appropriate handlers.
        """
        buffer = self.key_buffer
        retry = False
        while True:
            flush = False
            if retry:
                retry = False
            else:
                key = (yield)
                if key is _Flush:
                    flush = True
                else:
                    buffer.append(key)
            if buffer:
                matches = self._get_matches(buffer)
                if flush:
                    is_prefix_of_longer_match = False
                else:
                    is_prefix_of_longer_match = self._is_prefix_of_longer_match(buffer)
                eager_matches = [m for m in matches if m.eager()]
                if eager_matches:
                    matches = eager_matches
                    is_prefix_of_longer_match = False
                if not is_prefix_of_longer_match and matches:
                    self._call_handler(matches[-1], key_sequence=buffer[:])
                    del buffer[:]
                elif not is_prefix_of_longer_match and (not matches):
                    retry = True
                    found = False
                    for i in range(len(buffer), 0, -1):
                        matches = self._get_matches(buffer[:i])
                        if matches:
                            self._call_handler(matches[-1], key_sequence=buffer[:i])
                            del buffer[:i]
                            found = True
                            break
                    if not found:
                        del buffer[:1]

    def feed(self, key_press: KeyPress, first: bool=False) -> None:
        """
        Add a new :class:`KeyPress` to the input queue.
        (Don't forget to call `process_keys` in order to process the queue.)

        :param first: If true, insert before everything else.
        """
        if first:
            self.input_queue.appendleft(key_press)
        else:
            self.input_queue.append(key_press)

    def feed_multiple(self, key_presses: list[KeyPress], first: bool=False) -> None:
        """
        :param first: If true, insert before everything else.
        """
        if first:
            self.input_queue.extendleft(reversed(key_presses))
        else:
            self.input_queue.extend(key_presses)

    def process_keys(self) -> None:
        """
        Process all the keys in the `input_queue`.
        (To be called after `feed`.)

        Note: because of the `feed`/`process_keys` separation, it is
              possible to call `feed` from inside a key binding.
              This function keeps looping until the queue is empty.
        """
        app = get_app()

        def not_empty() -> bool:
            if app.is_done:
                return any((k for k in self.input_queue if k.key == Keys.CPRResponse))
            else:
                return bool(self.input_queue)

        def get_next() -> KeyPress:
            if app.is_done:
                cpr = [k for k in self.input_queue if k.key == Keys.CPRResponse][0]
                self.input_queue.remove(cpr)
                return cpr
            else:
                return self.input_queue.popleft()
        is_flush = False
        while not_empty():
            key_press = get_next()
            is_flush = key_press is _Flush
            is_cpr = key_press.key == Keys.CPRResponse
            if not is_flush and (not is_cpr):
                self.before_key_press.fire()
            try:
                self._process_coroutine.send(key_press)
            except Exception:
                self.reset()
                self.empty_queue()
                raise
            if not is_flush and (not is_cpr):
                self.after_key_press.fire()
        if not is_flush:
            self._start_timeout()

    def empty_queue(self) -> list[KeyPress]:
        """
        Empty the input queue. Return the unprocessed input.
        """
        key_presses = list(self.input_queue)
        self.input_queue.clear()
        key_presses = [k for k in key_presses if k.key != Keys.CPRResponse]
        return key_presses

    def _call_handler(self, handler: Binding, key_sequence: list[KeyPress]) -> None:
        app = get_app()
        was_recording_emacs = app.emacs_state.is_recording
        was_recording_vi = bool(app.vi_state.recording_register)
        was_temporary_navigation_mode = app.vi_state.temporary_navigation_mode
        arg = self.arg
        self.arg = None
        event = KeyPressEvent(weakref.ref(self), arg=arg, key_sequence=key_sequence, previous_key_sequence=self._previous_key_sequence, is_repeat=handler == self._previous_handler)
        if handler.save_before(event):
            event.app.current_buffer.save_to_undo_stack()
        from prompt_toolkit.buffer import EditReadOnlyBuffer
        try:
            handler.call(event)
            self._fix_vi_cursor_position(event)
        except EditReadOnlyBuffer:
            app.output.bell()
        if was_temporary_navigation_mode:
            self._leave_vi_temp_navigation_mode(event)
        self._previous_key_sequence = key_sequence
        self._previous_handler = handler
        if handler.record_in_macro():
            if app.emacs_state.is_recording and was_recording_emacs:
                recording = app.emacs_state.current_recording
                if recording is not None:
                    recording.extend(key_sequence)
            if app.vi_state.recording_register and was_recording_vi:
                for k in key_sequence:
                    app.vi_state.current_recording += k.data

    def _fix_vi_cursor_position(self, event: KeyPressEvent) -> None:
        """
        After every command, make sure that if we are in Vi navigation mode, we
        never put the cursor after the last character of a line. (Unless it's
        an empty line.)
        """
        app = event.app
        buff = app.current_buffer
        preferred_column = buff.preferred_column
        if vi_navigation_mode() and buff.document.is_cursor_at_the_end_of_line and (len(buff.document.current_line) > 0):
            buff.cursor_position -= 1
            buff.preferred_column = preferred_column

    def _leave_vi_temp_navigation_mode(self, event: KeyPressEvent) -> None:
        """
        If we're in Vi temporary navigation (normal) mode, return to
        insert/replace mode after executing one action.
        """
        app = event.app
        if app.editing_mode == EditingMode.VI:
            if app.vi_state.operator_func is None and self.arg is None:
                app.vi_state.temporary_navigation_mode = False

    def _start_timeout(self) -> None:
        """
        Start auto flush timeout. Similar to Vim's `timeoutlen` option.

        Start a background coroutine with a timer. When this timeout expires
        and no key was pressed in the meantime, we flush all data in the queue
        and call the appropriate key binding handlers.
        """
        app = get_app()
        timeout = app.timeoutlen
        if timeout is None:
            return

        async def wait() -> None:
            """Wait for timeout."""
            await sleep(timeout)
            if len(self.key_buffer) > 0:
                flush_keys()

        def flush_keys() -> None:
            """Flush keys."""
            self.feed(_Flush)
            self.process_keys()
        if self._flush_wait_task:
            self._flush_wait_task.cancel()
        self._flush_wait_task = app.create_background_task(wait())

    def send_sigint(self) -> None:
        """
        Send SIGINT. Immediately call the SIGINT key handler.
        """
        self.feed(KeyPress(key=Keys.SIGINT), first=True)
        self.process_keys()