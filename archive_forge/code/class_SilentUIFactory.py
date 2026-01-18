import warnings
class SilentUIFactory(NoninteractiveUIFactory):
    """A UI Factory which never prints anything.

    This is the default UI, if another one is never registered by a program
    using breezy, and it's also active for example inside 'brz serve'.

    Methods that try to read from the user raise an error; methods that do
    output do nothing.
    """

    def __init__(self):
        UIFactory.__init__(self)

    def note(self, msg):
        pass

    def get_username(self, prompt, **kwargs):
        return None

    def _make_output_stream_explicit(self, encoding, encoding_type):
        return NullOutputStream(encoding)

    def show_error(self, msg):
        pass

    def show_message(self, msg):
        pass

    def show_warning(self, msg):
        pass