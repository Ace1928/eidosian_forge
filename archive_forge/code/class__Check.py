import os
class _Check(_Strategy):

    def noexists(self, path, checker):
        checker.noexists(path)

    def check(self, path, checker):
        checker.check(path)