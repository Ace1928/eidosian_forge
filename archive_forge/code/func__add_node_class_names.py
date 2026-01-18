import sys
import os
import re
import warnings
import types
import unicodedata
def _add_node_class_names(names):
    """Save typing with dynamic assignments:"""
    for _name in names:
        setattr(GenericNodeVisitor, 'visit_' + _name, _call_default_visit)
        setattr(GenericNodeVisitor, 'depart_' + _name, _call_default_departure)
        setattr(SparseNodeVisitor, 'visit_' + _name, _nop)
        setattr(SparseNodeVisitor, 'depart_' + _name, _nop)