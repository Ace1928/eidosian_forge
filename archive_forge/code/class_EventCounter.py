import pytest
class EventCounter:

    def __init__(self, anim):
        self.n_start = 0
        self.n_progress = 0
        self.n_complete = 0
        anim.bind(on_start=self.on_start, on_progress=self.on_progress, on_complete=self.on_complete)

    def on_start(self, anim, widget):
        self.n_start += 1

    def on_progress(self, anim, widget, progress):
        self.n_progress += 1

    def on_complete(self, anim, widget):
        self.n_complete += 1

    def assert_(self, n_start, n_progress_greater_than_zero, n_complete):
        assert self.n_start == n_start
        if n_progress_greater_than_zero:
            assert self.n_progress > 0
        else:
            assert self.n_progress == 0
        assert self.n_complete == n_complete