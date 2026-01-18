import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def _get_node_tree(self):

    def show_node(node, items, shift):
        while node is not None:
            if node.is_text:
                items.append((shift, f'"{node.text}"'))
                node = node.next
                continue
            items.append((shift, f'({node.tagname}'))
            for k, v in node.get_attributes().items():
                items.append((shift, f"={k} '{v}'"))
            child = node.first_child
            if child:
                items = show_node(child, items, shift + 1)
            items.append((shift, f'){node.tagname}'))
            node = node.next
        return items
    shift = 0
    items = []
    items = show_node(self, items, shift)
    return items