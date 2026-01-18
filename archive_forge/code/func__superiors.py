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