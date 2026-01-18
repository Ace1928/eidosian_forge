def _should_use_paren(self, optval):
    if optval is not None:
        return optval
    return len(self.params) > 1