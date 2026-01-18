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
class PyElement(PyObj):

    def __init__(self, name=None, pyname=None, root=None, parent=''):
        PyObj.__init__(self, name, pyname, root)
        if parent:
            self.class_name = f'{leading_uppercase(parent)}_{self.name}'
        else:
            self.class_name = leading_uppercase(self.name)
        self.ref = None
        self.min = 1
        self.max = 1
        self.definition = None
        self.orig = None

    def undefined(self, cdict):
        try:
            mod, typ = self.type
            if not mod:
                cname = leading_uppercase(typ)
                if not cdict[cname].done:
                    return ([cdict[cname]], [])
        except ValueError:
            pass
        except TypeError:
            if isinstance(self.type, PyType):
                return self.type.undefined(cdict)
            elif isinstance(self.ref, tuple):
                pass
            else:
                cname = leading_uppercase(self.ref)
                if not cdict[cname].done:
                    return ([cdict[cname]], [])
        return ([], [])

    def _local_class(self, typ, cdict, child, target_namespace, ignore):
        if typ in cdict and (not cdict[typ].done):
            raise MissingPrerequisite(typ)
        else:
            self.orig = {'type': self.type}
            try:
                self.orig['superior'] = self.superior
            except AttributeError:
                self.orig['superior'] = []
            self.superior = [typ]
            req = self.class_definition(target_namespace, cdict, ignore)
            if not child:
                req = [req]
            if not hasattr(self, 'scoped'):
                cdict[self.name] = self
                cdict[self.name].done = True
                if child:
                    cdict[self.name].local = True
            self.type = (None, self.name)
        return req

    def _external_class(self, mod, typ, cdict, child, target_namespace, ignore):
        cname = self.root.modul[mod].factory(typ).__class__.__name__
        imp_name = f'{mod}.{cname}'
        if imp_name not in cdict:
            impo = pyelement_factory(imp_name, None, None)
            impo.properties = [_import_attrs(self.root.modul[mod], typ, self.root), []]
            impo.class_name = imp_name
            cdict[imp_name] = impo
            impo.done = True
            if child:
                impo.local = True
        self.superior = [imp_name]
        text = self.class_definition(target_namespace, cdict, ignore=ignore)
        return text

    def text(self, target_namespace, cdict, child=True, ignore=None):
        if ignore is None:
            ignore = []
        if child:
            text = []
        else:
            text = None
        req = []
        try:
            mod, typ = self.type
            if not mod:
                req = self._local_class(typ, cdict, child, target_namespace, ignore)
            else:
                text = self._external_class(mod, typ, cdict, child, target_namespace, ignore)
        except ValueError:
            if self.type:
                text = self.class_definition(target_namespace, cdict, ignore=ignore)
                if child:
                    self.local = True
                self.done = True
        except TypeError:
            if isinstance(self.type, PyObj):
                pyobj = self.type
                pyobj.name = self.name
                pyobj.pyname = self.pyname
                pyobj.class_name = self.class_name
                cdict[self.name] = pyobj
                return pyobj.text(target_namespace, cdict, ignore=ignore)
            elif isinstance(self.ref, tuple):
                mod, typ = self.ref
                if mod:
                    if verify_import(self.root.modul[mod], typ):
                        return (req, text)
                    else:
                        raise Exception(f"Import attempted on {typ} from {mod} module failed - wasn't there")
                elif not child:
                    self.superior = [typ]
                    text = self.class_definition(target_namespace, cdict, ignore=ignore)
            elif not cdict[class_pyify(self.ref)].done:
                raise MissingPrerequisite(self.ref)
        self.done = True
        return (req, text)