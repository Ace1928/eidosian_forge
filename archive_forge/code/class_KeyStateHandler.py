from pyglet import compat_platform
class KeyStateHandler:
    """Simple handler that tracks the state of keys on the keyboard. If a
    key is pressed then this handler holds a True value for it.
    If the window loses focus, all keys will be reset to False to avoid a
    "sticky" key state.

    For example::

        >>> win = window.Window
        >>> keyboard = key.KeyStateHandler()
        >>> win.push_handlers(keyboard)

        # Hold down the "up" arrow...

        >>> keyboard[key.UP]
        True
        >>> keyboard[key.DOWN]
        False

    """

    def __init__(self):
        self.data = {}

    def on_key_press(self, symbol, modifiers):
        self.data[symbol] = True

    def on_key_release(self, symbol, modifiers):
        self.data[symbol] = False

    def on_deactivate(self):
        self.data.clear()

    def __getitem__(self, key):
        return self.data.get(key, False)