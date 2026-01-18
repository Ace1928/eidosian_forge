from gi.repository import GLib
class _InputHook:

    def __init__(self, context):
        self._quit = False
        GLib.io_add_watch(context.fileno(), GLib.PRIORITY_DEFAULT, GLib.IO_IN, self.quit)

    def quit(self, *args, **kwargs):
        self._quit = True
        return False

    def run(self):
        context = GLib.MainContext.default()
        while not self._quit:
            context.iteration(True)