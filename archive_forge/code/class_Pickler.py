from pickle import (
class Pickler(_Pickler):

    def __init__(self, file, protocol=None, *, fix_imports=True, buffer_callback=None):
        super().__init__(file, protocol, fix_imports=fix_imports, buffer_callback=buffer_callback)
        self.dispatch = _Pickler.dispatch.copy()