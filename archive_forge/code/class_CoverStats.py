import re
import sys
import cgi
import os
import os.path
import urllib.parse
import cherrypy
class CoverStats(object):

    def __init__(self, coverage, root=None):
        self.coverage = coverage
        if root is None:
            root = os.path.dirname(cherrypy.__file__)
        self.root = root

    @cherrypy.expose
    def index(self):
        return TEMPLATE_FRAMESET % self.root.lower()

    @cherrypy.expose
    def menu(self, base='/', pct='50', showpct='', exclude='python\\d\\.\\d|test|tut\\d|tutorial'):
        base = base.lower().rstrip(os.sep)
        yield TEMPLATE_MENU
        yield (TEMPLATE_FORM % locals())
        yield "<div id='crumbs'>"
        path = ''
        atoms = base.split(os.sep)
        atoms.pop()
        for atom in atoms:
            path += atom + os.sep
            yield ("<a href='menu?base=%s&exclude=%s'>%s</a> %s" % (path, urllib.parse.quote_plus(exclude), atom, os.sep))
        yield '</div>'
        yield "<div id='tree'>"
        tree = get_tree(base, exclude, self.coverage)
        if not tree:
            yield '<p>No modules covered.</p>'
        else:
            for chunk in _show_branch(tree, base, '/', pct, showpct == 'checked', exclude, coverage=self.coverage):
                yield chunk
        yield '</div>'
        yield '</body></html>'

    def annotated_file(self, filename, statements, excluded, missing):
        with open(filename, 'r') as source:
            lines = source.readlines()
        buffer = []
        for lineno, line in enumerate(lines):
            lineno += 1
            line = line.strip('\n\r')
            empty_the_buffer = True
            if lineno in excluded:
                template = TEMPLATE_LOC_EXCLUDED
            elif lineno in missing:
                template = TEMPLATE_LOC_NOT_COVERED
            elif lineno in statements:
                template = TEMPLATE_LOC_COVERED
            else:
                empty_the_buffer = False
                buffer.append((lineno, line))
            if empty_the_buffer:
                for lno, pastline in buffer:
                    yield (template % (lno, cgi.escape(pastline)))
                buffer = []
                yield (template % (lineno, cgi.escape(line)))

    @cherrypy.expose
    def report(self, name):
        filename, statements, excluded, missing, _ = self.coverage.analysis2(name)
        pc = _percent(statements, missing)
        yield (TEMPLATE_COVERAGE % dict(name=os.path.basename(name), fullpath=name, pc=pc))
        yield '<table>\n'
        for line in self.annotated_file(filename, statements, excluded, missing):
            yield line
        yield '</table>'
        yield '</body>'
        yield '</html>'