def __cleanup_run(self):
    """Cleans up after a completed argument parsing run."""
    self.__stack = []
    assert not self.active()