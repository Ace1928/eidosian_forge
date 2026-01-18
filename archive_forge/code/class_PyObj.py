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
class PyObj:

    def __init__(self, name=None, pyname=None, root=None):
        self.name = name
        self.done = False
        self.local = True
        self.root = root
        self.superior = []
        self.value_type = ''
        self.properties = ([], [])
        self.abstract = False
        self.class_name = ''
        if pyname:
            self.pyname = pyname
        elif name:
            self.pyname = pyify(name)
        else:
            self.pyname = name
        self.type = None

    def child_spec(self, target_namespace, prop, mod, typ, lista):
        if mod:
            namesp = external_namespace(self.root.modul[mod])
            pkey = f'{{{namesp}}}{prop.name}'
            typ = f'{mod}.{typ}'
        else:
            pkey = f'{{{target_namespace}}}{prop.name}'
        if lista:
            return f"c_children['{pkey}'] = ('{prop.pyname}', [{typ}])"
        else:
            return f"c_children['{pkey}'] = ('{prop.pyname}', {typ})"

    def knamn(self, sup, cdict):
        cname = cdict[sup].class_name
        if not cname:
            namesp, tag = cdict[sup].name.split('.')
            if namesp:
                ctag = self.root.modul[namesp].factory(tag).__class__.__name__
                cname = f'{namesp}.{ctag}'
            else:
                cname = tag + '_'
        return cname

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

    def _superiors(self, cdict):
        imps = {}
        try:
            superior = self.superior
            sups = []
            for sup in superior:
                klass = self.knamn(sup, cdict)
                sups.append(klass)
                imps[klass] = []
                for cla in cdict[sup].properties[0]:
                    if cla.pyname and cla.pyname not in imps[klass]:
                        imps[klass].append(cla.pyname)
        except AttributeError:
            superior = []
            sups = []
        return (superior, sups, imps)

    def class_definition(self, target_namespace, cdict=None, ignore=None):
        line = []
        if self.root:
            if self.name not in [c.name for c in self.root.elems]:
                self.root.elems.append(self)
        superior, sups, imps = self._superiors(cdict)
        c_name = klass_namn(self)
        if not superior:
            line.append(f'class {c_name}(SamlBase):')
        else:
            line.append(f'class {c_name}({','.join(sups)}):')
        if hasattr(self, 'scoped'):
            pass
        else:
            line.append(f'{INDENT}"""The {target_namespace}:{self.name} element """')
        line.append('')
        line.append(f"{INDENT}c_tag = '{self.name}'")
        line.append(f'{INDENT}c_namespace = NAMESPACE')
        try:
            if self.value_type:
                if isinstance(self.value_type, str):
                    line.append(f"{INDENT}c_value_type = '{self.value_type}'")
                else:
                    line.append(f'{INDENT}c_value_type = {self.value_type}')
        except AttributeError:
            pass
        if not superior:
            for var, cps in CLASS_PROP:
                line.append(f'{INDENT}{var} = SamlBase.{var}{cps}')
        else:
            for sup in sups:
                for var, cps in CLASS_PROP:
                    line.append(f'{INDENT}{var} = {sup}.{var}{cps}')
        args, child, inh = self._do_properties(line, cdict, ignore, target_namespace)
        if child:
            line.append('{}c_child_order.extend([{}])'.format(INDENT, "'" + "', '".join(child) + "'"))
        if args:
            if inh:
                cname = self.knamn(self.superior[0], cdict)
                imps = {cname: [c.pyname for c in inh if c.pyname]}
            line.append('')
            line.extend(def_init(imps, args))
            line.extend(base_init(imps))
            line.extend(initialize(args))
        line.append('')
        if not self.abstract or not self.class_name.endswith('_'):
            line.append(f'def {pyify(self.class_name)}_from_string(xml_string):')
            line.append(f'{INDENT}return saml2.create_class_from_xml_string({self.class_name}, xml_string)')
            line.append('')
        self.done = True
        return '\n'.join(line)