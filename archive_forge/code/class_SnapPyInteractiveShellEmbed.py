from IPython.core.displayhook import DisplayHook
class SnapPyInteractiveShellEmbed(InteractiveShellEmbed):
    """
    An embedded IPython shell which can use a TkTerm for all of its
    input and output, including the output prompt.
    """
    readline_use = False
    autoindent_use = False
    colors_force = True
    separate_out = '\n'
    separate_in = ''

    def __init__(self, *args, **kwargs):
        super(InteractiveShellEmbed, self).__init__(*args, **kwargs)
        self.Completer.use_jedi = False
        self.magics_manager.magics['line']['colors']('LightBG')

    def _displayhook_class_default(self):
        return SnapPyPromptDisplayHook