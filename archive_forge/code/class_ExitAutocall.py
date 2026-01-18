class ExitAutocall(IPyAutocall):
    """An autocallable object which will be added to the user namespace so that
    exit, exit(), quit or quit() are all valid ways to close the shell."""
    rewrite = False

    def __call__(self):
        self._ip.ask_exit()