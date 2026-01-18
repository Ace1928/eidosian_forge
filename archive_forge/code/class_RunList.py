class RunList:
    """List of contiguous runs of values.

    A `RunList` is an efficient encoding of a sequence of values.  For
    example, the sequence ``aaaabbccccc`` is encoded as ``(4, a), (2, b),
    (5, c)``.  The class provides methods for modifying and querying the
    run list without needing to deal with the tricky cases of splitting and
    merging the run list entries.

    Run lists are used to represent formatted character data in pyglet.  A
    separate run list is maintained for each style attribute, for example,
    bold, italic, font size, and so on.  Unless you are overriding the
    document interfaces, the only interaction with run lists is via
    `RunIterator`.

    The length and ranges of a run list always refer to the character
    positions in the decoded list.  For example, in the above sequence,
    ``set_run(2, 5, 'x')`` would change the sequence to ``aaxxxbccccc``.
    """

    def __init__(self, size, initial):
        """Create a run list of the given size and a default value.

        :Parameters:
            `size` : int
                Number of characters to represent initially.
            `initial` : object
                The value of all characters in the run list.

        """
        self.runs = [_Run(initial, size)]

    def insert(self, pos, length):
        """Insert characters into the run list.

        The inserted characters will take on the value immediately preceding
        the insertion point (or the value of the first character, if `pos` is
        0).

        :Parameters:
            `pos` : int
                Insertion index
            `length` : int
                Number of characters to insert.

        """
        i = 0
        for run in self.runs:
            if i <= pos <= i + run.count:
                run.count += length
            i += run.count

    def delete(self, start, end):
        """Remove characters from the run list.

        :Parameters:
            `start` : int
                Starting index to remove from.
            `end` : int
                End index, exclusive.

        """
        i = 0
        for run in self.runs:
            if end - start == 0:
                break
            if i <= start <= i + run.count:
                trim = min(end - start, i + run.count - start)
                run.count -= trim
                end -= trim
            i += run.count
        self.runs = [r for r in self.runs if r.count > 0]
        if not self.runs:
            self.runs = [_Run(run.value, 0)]

    def set_run(self, start, end, value):
        """Set the value of a range of characters.

        :Parameters:
            `start` : int
                Start index of range.
            `end` : int
                End of range, exclusive.
            `value` : object
                Value to set over the range.

        """
        if end - start <= 0:
            return
        i = 0
        start_i = None
        start_trim = 0
        end_i = None
        end_trim = 0
        for run_i, run in enumerate(self.runs):
            count = run.count
            if i < start < i + count:
                start_i = run_i
                start_trim = start - i
            if i < end < i + count:
                end_i = run_i
                end_trim = end - i
            i += count
        if start_i is not None:
            run = self.runs[start_i]
            self.runs.insert(start_i, _Run(run.value, start_trim))
            run.count -= start_trim
            if end_i is not None:
                if end_i == start_i:
                    end_trim -= start_trim
                end_i += 1
        if end_i is not None:
            run = self.runs[end_i]
            self.runs.insert(end_i, _Run(run.value, end_trim))
            run.count -= end_trim
        i = 0
        for run in self.runs:
            if start <= i and i + run.count <= end:
                run.value = value
            i += run.count
        last_run = self.runs[0]
        for run in self.runs[1:]:
            if run.value == last_run.value:
                run.count += last_run.count
                last_run.count = 0
            last_run = run
        self.runs = [r for r in self.runs if r.count > 0]

    def __iter__(self):
        i = 0
        for run in self.runs:
            yield (i, i + run.count, run.value)
            i += run.count

    def get_run_iterator(self):
        """Get an extended iterator over the run list.

        :rtype: `RunIterator`
        """
        return RunIterator(self)

    def __getitem__(self, index):
        """Get the value at a character position.

        :Parameters:
            `index` : int
                Index of character.  Must be within range and non-negative.

        :rtype: object
        """
        i = 0
        for run in self.runs:
            if i <= index < i + run.count:
                return run.value
            i += run.count
        if index == i:
            return self.runs[-1].value
        raise IndexError

    def __repr__(self):
        return str(list(self))