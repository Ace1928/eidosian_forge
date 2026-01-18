from rdkit.ML.DecTree import DecTree, Tree
def SetQuantBounds(self, qBounds):
    self.qBounds = qBounds[:]
    self.nBounds = len(self.qBounds)