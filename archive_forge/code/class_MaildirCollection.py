import os
from twisted.spread import pb
class MaildirCollection(pb.Referenceable):

    def __init__(self, root):
        self.root = root

    def getSubFolders(self):
        return os.listdir(self.getRoot())
    remote_getSubFolders = getSubFolders

    def getSubFolder(self, name):
        if '/' in name or name[0] == '.':
            raise OSError('invalid name')
        return Maildir('.', os.path.join(self.getRoot(), name))
    remote_getSubFolder = getSubFolder