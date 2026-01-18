import sys
class VerboseFileWrapper(_ProgressBarBase):
    """A file wrapper with a progress bar.

    The file wrapper shows and advances a progress bar whenever the
    wrapped file's read method is called.
    """

    def read(self, *args, **kwargs):
        data = self._wrapped.read(*args, **kwargs)
        if data:
            self._display_progress_bar(len(data))
        elif self._show_progress:
            sys.stdout.write('\n')
        return data