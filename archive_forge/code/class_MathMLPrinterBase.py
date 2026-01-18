from __future__ import annotations
from typing import Any
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import \
from sympy.printing.pretty.pretty_symbology import greek_unicode
from sympy.printing.printer import Printer, print_function
from mpmath.libmp import prec_to_dps, repr_dps, to_str as mlib_to_str
class MathMLPrinterBase(Printer):
    """Contains common code required for MathMLContentPrinter and
    MathMLPresentationPrinter.
    """
    _default_settings: dict[str, Any] = {'order': None, 'encoding': 'utf-8', 'fold_frac_powers': False, 'fold_func_brackets': False, 'fold_short_frac': None, 'inv_trig_style': 'abbreviated', 'ln_notation': False, 'long_frac_ratio': None, 'mat_delim': '[', 'mat_symbol_style': 'plain', 'mul_symbol': None, 'root_notation': True, 'symbol_names': {}, 'mul_symbol_mathml_numbers': '&#xB7;'}

    def __init__(self, settings=None):
        Printer.__init__(self, settings)
        from xml.dom.minidom import Document, Text
        self.dom = Document()

        class RawText(Text):

            def writexml(self, writer, indent='', addindent='', newl=''):
                if self.data:
                    writer.write('{}{}{}'.format(indent, self.data, newl))

        def createRawTextNode(data):
            r = RawText()
            r.data = data
            r.ownerDocument = self.dom
            return r
        self.dom.createTextNode = createRawTextNode

    def doprint(self, expr):
        """
        Prints the expression as MathML.
        """
        mathML = Printer._print(self, expr)
        unistr = mathML.toxml()
        xmlbstr = unistr.encode('ascii', 'xmlcharrefreplace')
        res = xmlbstr.decode()
        return res

    def apply_patch(self):
        from xml.dom.minidom import Element, Text, Node, _write_data

        def writexml(self, writer, indent='', addindent='', newl=''):
            writer.write(indent + '<' + self.tagName)
            attrs = self._get_attributes()
            a_names = list(attrs.keys())
            a_names.sort()
            for a_name in a_names:
                writer.write(' %s="' % a_name)
                _write_data(writer, attrs[a_name].value)
                writer.write('"')
            if self.childNodes:
                writer.write('>')
                if len(self.childNodes) == 1 and self.childNodes[0].nodeType == Node.TEXT_NODE:
                    self.childNodes[0].writexml(writer, '', '', '')
                else:
                    writer.write(newl)
                    for node in self.childNodes:
                        node.writexml(writer, indent + addindent, addindent, newl)
                    writer.write(indent)
                writer.write('</%s>%s' % (self.tagName, newl))
            else:
                writer.write('/>%s' % newl)
        self._Element_writexml_old = Element.writexml
        Element.writexml = writexml

        def writexml(self, writer, indent='', addindent='', newl=''):
            _write_data(writer, '%s%s%s' % (indent, self.data, newl))
        self._Text_writexml_old = Text.writexml
        Text.writexml = writexml

    def restore_patch(self):
        from xml.dom.minidom import Element, Text
        Element.writexml = self._Element_writexml_old
        Text.writexml = self._Text_writexml_old