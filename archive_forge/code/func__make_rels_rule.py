import codecs
import copy
import json
import os
from urllib.parse import unquote
import bs4
from . import mf2_classes
from .dom_helpers import get_children
from .mf_helpers import unordered_list
def _make_rels_rule(old_rels, new_classes, html_parser):
    """Builds a rule for augmenting an mf1 rel with its mf2 class equivalent(s)."""

    def f(child, **kwargs):
        child_rels = child.get('rel', [])
        child_classes = child.get('class', [])[:]
        if all((r in child_rels for r in old_rels)):
            if 'tag' in old_rels:
                _rel_tag_to_category_rule(child, html_parser, **kwargs)
            else:
                child_classes.extend([cl for cl in new_classes if cl not in child_classes])
                child['class'] = child_classes
    return f