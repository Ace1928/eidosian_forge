import os
from twisted.spread import pb
def _isSubFolder(self, name):
    return not os.path.isdir(os.path.join(self.rootDirectory, name)) or not os.path.isfile(os.path.join(self.rootDirectory, name, 'maildirfolder'))