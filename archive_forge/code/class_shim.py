import sys
import os
class shim:

    def __enter__(self):
        insert_shim()

    def __exit__(self, exc, value, tb):
        _remove_shim()