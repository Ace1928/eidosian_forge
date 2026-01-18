import sys
class VerboseIteratorWrapper(_ProgressBarBase):
    """An iterator wrapper with a progress bar.

    The iterator wrapper shows and advances a progress bar whenever the
    wrapped data is consumed from the iterator.

    :note: Use only with iterator that yield strings.
    """

    def __iter__(self):
        return self

    def next(self):
        try:
            data = next(self._wrapped)
            self._display_progress_bar(len(data))
            return data
        except StopIteration:
            if self._show_progress:
                sys.stdout.write('\n')
            raise
    __next__ = next