from __future__ import absolute_import, division, unicode_literals
from six import text_type
from collections import OrderedDict
from lxml import etree
from ..treebuilders.etree import tag_regexp
from . import base
from .. import _ihatexml
class FragmentWrapper(object):

    def __init__(self, fragment_root, obj):
        self.root_node = fragment_root
        self.obj = obj
        if hasattr(self.obj, 'text'):
            self.text = ensure_str(self.obj.text)
        else:
            self.text = None
        if hasattr(self.obj, 'tail'):
            self.tail = ensure_str(self.obj.tail)
        else:
            self.tail = None

    def __getattr__(self, name):
        return getattr(self.obj, name)

    def getnext(self):
        siblings = self.root_node.children
        idx = siblings.index(self)
        if idx < len(siblings) - 1:
            return siblings[idx + 1]
        else:
            return None

    def __getitem__(self, key):
        return self.obj[key]

    def __bool__(self):
        return bool(self.obj)

    def getparent(self):
        return None

    def __str__(self):
        return str(self.obj)

    def __unicode__(self):
        return str(self.obj)

    def __len__(self):
        return len(self.obj)