import codecs
import copy
import json
import os
from urllib.parse import unquote
import bs4
from . import mf2_classes
from .dom_helpers import get_children
from .mf_helpers import unordered_list
def _get_rules(old_root, html_parser):
    """for given mf1 root get the rules as a list of functions to act on children"""
    class_rules = [_make_classes_rule(old_classes.split(), new_classes) for old_classes, new_classes in _CLASSIC_MAP[old_root].get('properties', {}).items()]
    rel_rules = [_make_rels_rule(old_rels.split(), new_classes, html_parser) for old_rels, new_classes in _CLASSIC_MAP[old_root].get('rels', {}).items()]
    return class_rules + rel_rules