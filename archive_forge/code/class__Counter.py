from reportlab.rl_config import register_reset
class _Counter:
    """Private class used by Sequencer.  Each counter
    knows its format, and the IDs of anything it
    resets, as well as its value. Starts at zero
    and increments just before you get the new value,
    so that it is still 'Chapter 5' and not 'Chapter 6'
    when you print 'Figure 5.1'"""

    def __init__(self):
        self._base = 0
        self._value = self._base
        self._formatter = _format_123
        self._resets = []

    def setFormatter(self, formatFunc):
        self._formatter = formatFunc

    def reset(self, value=None):
        if value:
            self._value = value
        else:
            self._value = self._base

    def next(self):
        self._value += 1
        v = self._value
        for counter in self._resets:
            counter.reset()
        return v
    __next__ = next

    def _this(self):
        return self._value

    def nextf(self):
        """Returns next value formatted"""
        return self._formatter(next(self))

    def thisf(self):
        return self._formatter(self._this())

    def chain(self, otherCounter):
        if not otherCounter in self._resets:
            self._resets.append(otherCounter)