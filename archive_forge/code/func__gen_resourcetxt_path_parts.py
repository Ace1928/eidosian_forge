import unittest
import os
import contextlib
import importlib_resources as resources
def _gen_resourcetxt_path_parts(self):
    """Yield various names of a text file in anchor02, each in a subTest"""
    for path_parts in (('subdirectory', 'subsubdir', 'resource.txt'), ('subdirectory/subsubdir/resource.txt',), ('subdirectory/subsubdir', 'resource.txt')):
        with self.subTest(path_parts=path_parts):
            yield path_parts