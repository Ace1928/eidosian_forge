from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class Declarator(object):

    def generate(self, with_semicolon=True):
        tp_lines, tp_decl = self.get_decl_pair()
        tp_lines = list(tp_lines)
        for line in tp_lines[:-1]:
            yield line
        sc = ';' if with_semicolon else ''
        if tp_decl is None:
            yield ('%s%s' % (tp_lines[-1], sc))
        else:
            yield ('%s %s%s' % (tp_lines[-1], tp_decl, sc))

    def get_decl_pair(self):
        """Return a tuple ``(type_lines, rhs)``.

        *type_lines* is a non-empty list of lines (most often just a
        single one) describing the type of this declarator. *rhs* is the right-
        hand side that actually contains the function/array/constness notation
        making up the bulk of the declarator syntax.
        """

    def inline(self):
        """Return the declarator as a single line."""
        tp_lines, tp_decl = self.get_decl_pair()
        tp_lines = ' '.join(tp_lines)
        if tp_decl is None:
            return tp_lines
        else:
            return '%s %s' % (tp_lines, tp_decl)