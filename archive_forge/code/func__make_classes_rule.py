import codecs
import copy
import json
import os
from urllib.parse import unquote
import bs4
from . import mf2_classes
from .dom_helpers import get_children
from .mf_helpers import unordered_list
def _make_classes_rule(old_classes, new_classes):
    """Builds a rule for augmenting an mf1 class with its mf2
    equivalent(s).
    """

    def f(child, **kwargs):
        child_original = child.original or copy.copy(child)
        child_classes = child.get('class', [])[:]
        if all((cl in child_classes for cl in old_classes)):
            child_classes.extend([cl for cl in new_classes if cl not in child_classes])
            child['class'] = child_classes
            if mf2_classes.has_embedded_class(child_classes) and child.original is None:
                child.original = child_original
    return f