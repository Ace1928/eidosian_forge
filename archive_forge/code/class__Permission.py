import os
class _Permission(_Rule):
    name = '*'

    def __init__(self, perm, owner, dir):
        self.perm_spec = read_perm_spec(perm)
        self.owner = owner
        self.dir = dir

    def check(self, path):
        return mode_diff(path, self.perm_spec)

    def fix(self, path):
        set_mode(path, self.perm_spec)