import sys, re
class SimpleUnicodeVisitor(object):
    """ recursive visitor to write unicode. """

    def __init__(self, write, indent=0, curindent=0, shortempty=True):
        self.write = write
        self.cache = {}
        self.visited = {}
        self.indent = indent
        self.curindent = curindent
        self.parents = []
        self.shortempty = shortempty

    def visit(self, node):
        """ dispatcher on node's class/bases name. """
        cls = node.__class__
        try:
            visitmethod = self.cache[cls]
        except KeyError:
            for subclass in cls.__mro__:
                visitmethod = getattr(self, subclass.__name__, None)
                if visitmethod is not None:
                    break
            else:
                visitmethod = self.__object
            self.cache[cls] = visitmethod
        visitmethod(node)

    def __object(self, obj):
        self.write(escape(unicode(obj)))

    def raw(self, obj):
        self.write(obj.uniobj)

    def list(self, obj):
        assert id(obj) not in self.visited
        self.visited[id(obj)] = 1
        for elem in obj:
            self.visit(elem)

    def Tag(self, tag):
        assert id(tag) not in self.visited
        try:
            tag.parent = self.parents[-1]
        except IndexError:
            tag.parent = None
        self.visited[id(tag)] = 1
        tagname = getattr(tag, 'xmlname', tag.__class__.__name__)
        if self.curindent and (not self._isinline(tagname)):
            self.write('\n' + u(' ') * self.curindent)
        if tag:
            self.curindent += self.indent
            self.write(u('<%s%s>') % (tagname, self.attributes(tag)))
            self.parents.append(tag)
            for x in tag:
                self.visit(x)
            self.parents.pop()
            self.write(u('</%s>') % tagname)
            self.curindent -= self.indent
        else:
            nameattr = tagname + self.attributes(tag)
            if self._issingleton(tagname):
                self.write(u('<%s/>') % (nameattr,))
            else:
                self.write(u('<%s></%s>') % (nameattr, tagname))

    def attributes(self, tag):
        attrlist = dir(tag.attr)
        attrlist.sort()
        l = []
        for name in attrlist:
            res = self.repr_attribute(tag.attr, name)
            if res is not None:
                l.append(res)
        l.extend(self.getstyle(tag))
        return u('').join(l)

    def repr_attribute(self, attrs, name):
        if name[:2] != '__':
            value = getattr(attrs, name)
            if name.endswith('_'):
                name = name[:-1]
            if isinstance(value, raw):
                insert = value.uniobj
            else:
                insert = escape(unicode(value))
            return ' %s="%s"' % (name, insert)

    def getstyle(self, tag):
        """ return attribute list suitable for styling. """
        try:
            styledict = tag.style.__dict__
        except AttributeError:
            return []
        else:
            stylelist = [x + ': ' + y for x, y in styledict.items()]
            return [u(' style="%s"') % u('; ').join(stylelist)]

    def _issingleton(self, tagname):
        """can (and will) be overridden in subclasses"""
        return self.shortempty

    def _isinline(self, tagname):
        """can (and will) be overridden in subclasses"""
        return False