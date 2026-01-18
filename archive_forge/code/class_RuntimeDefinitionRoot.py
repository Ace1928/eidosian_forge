import json
import os
import sys
class RuntimeDefinitionRoot(object):
    """Abstraction that allows us to access files in the runtime definiton."""

    def __init__(self, path):
        self.root = path

    def read_file(self, *name):
        with open(os.path.join(self.root, *name)) as src:
            return src.read()