from lib2to3 import fixer_base
from lib2to3.pygram import token
from lib2to3.fixer_util import Name, syms, Node, Leaf, touch_import, Call, \
 If an INDENT is followed by a thing with a prefix then nuke the prefix
        Otherwise we get in trouble when removing __metaclass__ at suite start
    