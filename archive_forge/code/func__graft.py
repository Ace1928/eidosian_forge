import re
import sys
import cgi
import os
import os.path
import urllib.parse
import cherrypy
def _graft(path, tree):
    d = tree
    p = path
    atoms = []
    while True:
        p, tail = os.path.split(p)
        if not tail:
            break
        atoms.append(tail)
    atoms.append(p)
    if p != '/':
        atoms.append('/')
    atoms.reverse()
    for node in atoms:
        if node:
            d = d.setdefault(node, {})