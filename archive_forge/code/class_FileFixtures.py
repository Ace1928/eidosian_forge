import os
from libcloud.utils.py3 import u
class FileFixtures:

    def __init__(self, fixtures_type, sub_dir=''):
        script_dir = os.path.abspath(os.path.split(__file__)[0])
        self.root = os.path.join(script_dir, FIXTURES_ROOT[fixtures_type], sub_dir)

    def load(self, file):
        path = os.path.join(self.root, file)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as fh:
                content = fh.read()
            return u(content)
        else:
            raise OSError(path)