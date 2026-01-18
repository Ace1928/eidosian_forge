import sys, re
class HtmlTag(Tag):

    def unicode(self, indent=2):
        l = []
        HtmlVisitor(l.append, indent, shortempty=False).visit(self)
        return u('').join(l)