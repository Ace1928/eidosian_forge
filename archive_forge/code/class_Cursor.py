from typing import List, Union
class Cursor:

    def __init__(self, cid: int) -> None:
        self.cid = cid
        self.max_idle = 0
        self.count = 0

    def build_args(self):
        args = [str(self.cid)]
        if self.max_idle:
            args += ['MAXIDLE', str(self.max_idle)]
        if self.count:
            args += ['COUNT', str(self.count)]
        return args