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