import sys
import os
import contextlib
from ..backend import KeyringBackend
from ..credentials import SimpleCredential
from ..errors import PasswordDeleteError
from ..errors import PasswordSetError, InitError, KeyringLocked
from .._compat import properties
def _migrate(self, service):
    old_folder = 'Python'
    entry_list = []
    if self.iface.hasFolder(self.handle, old_folder, self.appid):
        entry_list = self.iface.readPasswordList(self.handle, old_folder, '*@*', self.appid)
        for entry in entry_list.items():
            key = entry[0]
            password = entry[1]
            username, service = key.rsplit('@', 1)
            ret = self.iface.writePassword(self.handle, service, username, password, self.appid)
            if ret == 0:
                self.iface.removeEntry(self.handle, old_folder, key, self.appid)
        entry_list = self.iface.readPasswordList(self.handle, old_folder, '*', self.appid)
        if not entry_list:
            self.iface.removeFolder(self.handle, old_folder, self.appid)