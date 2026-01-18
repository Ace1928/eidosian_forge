from __future__ import unicode_literals
from prompt_toolkit.buffer import EditReadOnlyBuffer
from prompt_toolkit.filters.cli import ViNavigationMode
from prompt_toolkit.keys import Keys, Key
from prompt_toolkit.utils import Event
from .registry import BaseRegistry
from collections import deque
from six.moves import range
import weakref
import six
class InputProcessor(object):
    """
    Statemachine that receives :class:`KeyPress` instances and according to the
    key bindings in the given :class:`Registry`, calls the matching handlers.

    ::

        p = InputProcessor(registry)

        # Send keys into the processor.
        p.feed(KeyPress(Keys.ControlX, '\x18'))
        p.feed(KeyPress(Keys.ControlC, '\x03')

        # Process all the keys in the queue.
        p.process_keys()

        # Now the ControlX-ControlC callback will be called if this sequence is
        # registered in the registry.

    :param registry: `BaseRegistry` instance.
    :param cli_ref: weakref to `CommandLineInterface`.
    """

    def __init__(self, registry, cli_ref):
        assert isinstance(registry, BaseRegistry)
        self._registry = registry
        self._cli_ref = cli_ref
        self.beforeKeyPress = Event(self)
        self.afterKeyPress = Event(self)
        self.input_queue = deque()
        self.key_buffer = []
        self.record_macro = False
        self.macro = []
        self.reset()

    def reset(self):
        self._previous_key_sequence = None
        self._previous_handler = None
        self._process_coroutine = self._process()
        self._process_coroutine.send(None)
        self.arg = None

    def start_macro(self):
        """ Start recording macro. """
        self.record_macro = True
        self.macro = []

    def end_macro(self):
        """ End recording macro. """
        self.record_macro = False

    def call_macro(self):
        for k in self.macro:
            self.feed(k)

    def _get_matches(self, key_presses):
        """
        For a list of :class:`KeyPress` instances. Give the matching handlers
        that would handle this.
        """
        keys = tuple((k.key for k in key_presses))
        cli = self._cli_ref()
        return [b for b in self._registry.get_bindings_for_keys(keys) if b.filter(cli)]

    def _is_prefix_of_longer_match(self, key_presses):
        """
        For a list of :class:`KeyPress` instances. Return True if there is any
        handler that is bound to a suffix of this keys.
        """
        keys = tuple((k.key for k in key_presses))
        cli = self._cli_ref()
        filters = set((b.filter for b in self._registry.get_bindings_starting_with_keys(keys)))
        return any((f(cli) for f in filters))

    def _process(self):
        """
        Coroutine implementing the key match algorithm. Key strokes are sent
        into this generator, and it calls the appropriate handlers.
        """
        buffer = self.key_buffer
        retry = False
        while True:
            if retry:
                retry = False
            else:
                buffer.append((yield))
            if buffer:
                is_prefix_of_longer_match = self._is_prefix_of_longer_match(buffer)
                matches = self._get_matches(buffer)
                eager_matches = [m for m in matches if m.eager(self._cli_ref())]
                if eager_matches:
                    matches = eager_matches
                    is_prefix_of_longer_match = False
                if not is_prefix_of_longer_match and matches:
                    self._call_handler(matches[-1], key_sequence=buffer)
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

    def feed(self, key_press):
        """
        Add a new :class:`KeyPress` to the input queue.
        (Don't forget to call `process_keys` in order to process the queue.)
        """
        assert isinstance(key_press, KeyPress)
        self.input_queue.append(key_press)

    def process_keys(self):
        """
        Process all the keys in the `input_queue`.
        (To be called after `feed`.)

        Note: because of the `feed`/`process_keys` separation, it is
              possible to call `feed` from inside a key binding.
              This function keeps looping until the queue is empty.
        """
        while self.input_queue:
            key_press = self.input_queue.popleft()
            if key_press.key != Keys.CPRResponse:
                self.beforeKeyPress.fire()
            self._process_coroutine.send(key_press)
            if key_press.key != Keys.CPRResponse:
                self.afterKeyPress.fire()
        cli = self._cli_ref()
        if cli:
            cli.invalidate()

    def _call_handler(self, handler, key_sequence=None):
        was_recording = self.record_macro
        arg = self.arg
        self.arg = None
        event = KeyPressEvent(weakref.ref(self), arg=arg, key_sequence=key_sequence, previous_key_sequence=self._previous_key_sequence, is_repeat=handler == self._previous_handler)
        cli = event.cli
        if handler.save_before(event) and cli:
            cli.current_buffer.save_to_undo_stack()
        try:
            handler.call(event)
            self._fix_vi_cursor_position(event)
        except EditReadOnlyBuffer:
            pass
        self._previous_key_sequence = key_sequence
        self._previous_handler = handler
        if self.record_macro and was_recording:
            self.macro.extend(key_sequence)

    def _fix_vi_cursor_position(self, event):
        """
        After every command, make sure that if we are in Vi navigation mode, we
        never put the cursor after the last character of a line. (Unless it's
        an empty line.)
        """
        cli = self._cli_ref()
        if cli:
            buff = cli.current_buffer
            preferred_column = buff.preferred_column
            if ViNavigationMode()(event.cli) and buff.document.is_cursor_at_the_end_of_line and (len(buff.document.current_line) > 0):
                buff.cursor_position -= 1
                buff.preferred_column = preferred_column