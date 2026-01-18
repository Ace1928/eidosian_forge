import os
import time
class ProgressPhase:
    """Update progress object with the current phase"""

    def __init__(self, message, total, pb):
        object.__init__(self)
        self.pb = pb
        self.message = message
        self.total = total
        self.cur_phase = None

    def next_phase(self):
        if self.cur_phase is None:
            self.cur_phase = 0
        else:
            self.cur_phase += 1
        self.pb.update(self.message, self.cur_phase, self.total)