import sys
class _ProgressBarBase(object):
    """A progress bar provider for a wrapped obect.

    Base abstract class used by specific class wrapper to show
    a progress bar when the wrapped object are consumed.

    :param wrapped: Object to wrap that hold data to be consumed.
    :param totalsize: The total size of the data in the wrapped object.

    :note: The progress will be displayed only if sys.stdout is a tty.
    """

    def __init__(self, wrapped, totalsize):
        self._wrapped = wrapped
        self._totalsize = float(totalsize)
        self._show_progress = sys.stdout.isatty() and self._totalsize != 0
        self._percent = 0

    def _display_progress_bar(self, size_read):
        if self._show_progress:
            self._percent += size_read / self._totalsize
            sys.stdout.write('\r[{0:<30}] {1:.0%}'.format('=' * int(round(self._percent * 29)) + '>', self._percent))
            sys.stdout.flush()

    def __getattr__(self, attr):
        return getattr(self._wrapped, attr)