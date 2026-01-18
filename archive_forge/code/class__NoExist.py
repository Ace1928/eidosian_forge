import os
class _NoExist(_Rule):
    name = 'noexist'

    def __init__(self, path):
        self.path = path

    def check(self, path):
        return ['Path %s should not exist' % path]

    def noexists(self, path):
        return []

    def fix(self, path):
        pass