import logging
import re
class Extendlist(list):

    def __setitem__(self, i, v):
        try:
            list.__setitem__(self, i, v)
        except IndexError:
            if len(self) == i:
                self.append(v)
            else:
                raise