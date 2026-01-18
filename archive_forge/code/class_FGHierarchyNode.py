import os
import re
import weakref
from rdkit import Chem, RDConfig
class FGHierarchyNode(object):
    children = None
    name = ''
    label = ''
    pattern = None
    smarts = ''
    rxnSmarts = ''
    parent = None
    removalReaction = None

    def __init__(self, name, patt, smarts='', label='', rxnSmarts='', parent=None):
        self.name = name
        self.pattern = patt
        if parent:
            self.parent = weakref.ref(parent)
        self.label = label
        self.smarts = smarts
        self.children = []
        self.rxnSmarts = rxnSmarts

    def __len__(self):
        res = 1
        for child in self.children:
            res += len(child)
        return res