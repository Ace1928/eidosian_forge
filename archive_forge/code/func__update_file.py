import os
from breezy import tests
def _update_file(self, path, text, checkin=True):
    """append text to file 'path' and check it in"""
    with open(path, 'a') as f:
        f.write(text)
    if checkin:
        self.run_bzr(['ci', path, '-m', '"' + path + '"'])