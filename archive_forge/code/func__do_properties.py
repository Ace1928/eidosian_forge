import errno
import getopt
import importlib
import re
import sys
import time
import types
from xml.etree import ElementTree as ElementTree
import saml2
from saml2 import SamlBase
def _do_properties(self, line, cdict, ignore, target_namespace):
    args = []
    child = []
    try:
        own, inh = self.properties
    except AttributeError:
        own, inh = ([], [])
    for prop in own:
        if isinstance(prop, PyAttribute):
            line.append(f"{INDENT}c_attributes['{prop.name}'] = {prop.spec()}")
            if prop.fixed:
                args.append((prop.pyname, prop.fixed, None))
            elif prop.default:
                args.append((prop.pyname, prop.pyname, prop.default))
            else:
                args.append((prop.pyname, prop.pyname, None))
        elif isinstance(prop, PyElement):
            mod, cname = _mod_cname(prop, cdict)
            if prop.max == 'unbounded':
                lista = True
                pmax = 0
            else:
                pmax = int(prop.max)
                lista = False
            if prop.name in ignore:
                pass
            else:
                line.append(f'{INDENT}{self.child_spec(target_namespace, prop, mod, cname, lista)}')
            pmin = int(getattr(prop, 'min', 1))
            if pmax == 1 and pmin == 1:
                pass
            elif prop.max == 'unbounded':
                line.append(f"""{INDENT}c_cardinality['{prop.pyname}'] = {{"min":{pmin}}}""")
            else:
                line.append('%sc_cardinality[\'%s\'] = {"min":%s, "max":%d}' % (INDENT, prop.pyname, pmin, pmax))
            child.append(prop.pyname)
            if lista:
                args.append((prop.pyname, f'{prop.pyname} or []', None))
            else:
                args.append((prop.pyname, prop.pyname, None))
    return (args, child, inh)