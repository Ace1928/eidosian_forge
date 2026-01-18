import numpy as np
def addFields(self, **fields):
    for f in fields:
        if f not in self.fields:
            self.fields[f] = None