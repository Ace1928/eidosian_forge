from __future__ import division
import datetime
import math
class BouncingBar(Bar):

    def update(self, pbar, width):
        """Updates the progress bar and its subcomponents."""
        left, marker, right = (format_updatable(i, pbar) for i in (self.left, self.marker, self.right))
        width -= len(left) + len(right)
        if pbar.finished:
            return '%s%s%s' % (left, width * marker, right)
        position = int(pbar.currval % (width * 2 - 1))
        if position > width:
            position = width * 2 - position
        lpad = self.fill * (position - 1)
        rpad = self.fill * (width - len(marker) - len(lpad))
        if not self.fill_left:
            rpad, lpad = (lpad, rpad)
        return '%s%s%s%s%s' % (left, lpad, marker, rpad, right)