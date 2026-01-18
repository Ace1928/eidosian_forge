import logging
class ReSTStyle(BaseStyle):

    def __init__(self, doc, indent_width=2):
        BaseStyle.__init__(self, doc, indent_width)
        self.do_p = True
        self.a_href = None
        self.list_depth = 0

    def new_paragraph(self):
        self.doc.write('\n\n%s' % self.spaces())

    def new_line(self):
        self.doc.write('\n%s' % self.spaces())

    def _start_inline(self, markup):
        try:
            last_write = self.doc.peek_write()
        except IndexError:
            pass
        else:
            if last_write in ('*', '**') and markup in ('*', '**'):
                self.doc.write(' ')
        self.doc.write(markup)

    def _end_inline(self, markup):
        last_write = self.doc.pop_write()
        if last_write == markup:
            return
        self.doc.push_write(last_write)
        self.doc.write(markup)

    def start_bold(self, attrs=None):
        self._start_inline('**')

    def end_bold(self):
        self._end_inline('**')

    def start_b(self, attrs=None):
        self.doc.do_translation = True
        self.start_bold(attrs)

    def end_b(self):
        self.doc.do_translation = False
        self.end_bold()

    def bold(self, s):
        if s:
            self.start_bold()
            self.doc.write(s)
            self.end_bold()

    def ref(self, title, link=None):
        if link is None:
            link = title
        self.doc.write(f':doc:`{title} <{link}>`')

    def _heading(self, s, border_char):
        border = border_char * len(s)
        self.new_paragraph()
        self.doc.write(f'{border}\n{s}\n{border}')
        self.new_paragraph()

    def h1(self, s):
        self._heading(s, '*')

    def h2(self, s):
        self._heading(s, '=')

    def h3(self, s):
        self._heading(s, '-')

    def start_italics(self, attrs=None):
        self._start_inline('*')

    def end_italics(self):
        self._end_inline('*')

    def italics(self, s):
        if s:
            self.start_italics()
            self.doc.write(s)
            self.end_italics()

    def start_p(self, attrs=None):
        if self.do_p:
            self.doc.write('\n\n%s' % self.spaces())

    def end_p(self):
        if self.do_p:
            self.doc.write('\n\n%s' % self.spaces())

    def start_code(self, attrs=None):
        self.doc.do_translation = True
        self.add_trailing_space_to_previous_write()
        self._start_inline('``')

    def end_code(self):
        self.doc.do_translation = False
        self._end_inline('``')

    def code(self, s):
        if s:
            self.start_code()
            self.doc.write(s)
            self.end_code()

    def start_note(self, attrs=None):
        self.new_paragraph()
        self.doc.write('.. note::')
        self.indent()
        self.new_paragraph()

    def end_note(self):
        self.dedent()
        self.new_paragraph()

    def start_important(self, attrs=None):
        self.new_paragraph()
        self.doc.write('.. warning::')
        self.indent()
        self.new_paragraph()

    def end_important(self):
        self.dedent()
        self.new_paragraph()

    def start_danger(self, attrs=None):
        self.new_paragraph()
        self.doc.write('.. danger::')
        self.indent()
        self.new_paragraph()

    def end_danger(self):
        self.dedent()
        self.new_paragraph()

    def start_a(self, attrs=None):
        self.add_trailing_space_to_previous_write()
        if attrs:
            for attr_key, attr_value in attrs:
                if attr_key == 'href':
                    self.a_href = attr_value.strip()
                    self.doc.write('`')
        else:
            self.doc.write(' ')
        self.doc.do_translation = True

    def link_target_definition(self, refname, link):
        self.doc.writeln(f'.. _{refname}: {link}')

    def sphinx_reference_label(self, label, text=None):
        if text is None:
            text = label
        if self.doc.target == 'html':
            self.doc.write(f':ref:`{text} <{label}>`')
        else:
            self.doc.write(text)

    def _clean_link_text(self):
        doc = self.doc
        last_write = doc.pop_write()
        while not last_write.startswith('`'):
            last_write = doc.pop_write() + last_write
        if last_write != '':
            if last_write.startswith('` '):
                last_write = f'`{last_write[1:].lstrip(' ')}'
            doc.push_write(last_write)

    def end_a(self, next_child=None):
        self.doc.do_translation = False
        if self.a_href:
            self._clean_link_text()
            last_write = self.doc.pop_write()
            last_write = last_write.rstrip(' ')
            if last_write and last_write != '`':
                if ':' in last_write:
                    last_write = last_write.replace(':', '\\:')
                self.doc.push_write(last_write)
                self.doc.push_write(' <%s>`__' % self.a_href)
            elif last_write == '`':
                self.doc.push_write('`<%s>`__' % self.a_href)
            else:
                self.doc.push_write(self.a_href)
                self.doc.hrefs[self.a_href] = self.a_href
                self.doc.write('`__')
            self.a_href = None

    def start_i(self, attrs=None):
        self.doc.do_translation = True
        self.start_italics()

    def end_i(self):
        self.doc.do_translation = False
        self.end_italics()

    def start_li(self, attrs=None):
        self.new_line()
        self.do_p = False
        self.doc.write('* ')

    def end_li(self):
        self.do_p = True
        self.new_line()

    def li(self, s):
        if s:
            self.start_li()
            self.doc.writeln(s)
            self.end_li()

    def start_ul(self, attrs=None):
        if self.list_depth != 0:
            self.indent()
        self.list_depth += 1
        self.new_paragraph()

    def end_ul(self):
        self.list_depth -= 1
        if self.list_depth != 0:
            self.dedent()
        self.new_paragraph()

    def start_ol(self, attrs=None):
        if self.list_depth != 0:
            self.indent()
        self.list_depth += 1
        self.new_paragraph()

    def end_ol(self):
        self.list_depth -= 1
        if self.list_depth != 0:
            self.dedent()
        self.new_paragraph()

    def start_examples(self, attrs=None):
        self.doc.keep_data = False

    def end_examples(self):
        self.doc.keep_data = True

    def start_fullname(self, attrs=None):
        self.doc.keep_data = False

    def end_fullname(self):
        self.doc.keep_data = True

    def start_codeblock(self, attrs=None):
        self.doc.write('::')
        self.indent()
        self.new_paragraph()

    def end_codeblock(self):
        self.dedent()
        self.new_paragraph()

    def codeblock(self, code):
        """
        Literal code blocks are introduced by ending a paragraph with
        the special marker ::.  The literal block must be indented
        (and, like all paragraphs, separated from the surrounding
        ones by blank lines).
        """
        self.start_codeblock()
        self.doc.writeln(code)
        self.end_codeblock()

    def toctree(self):
        if self.doc.target == 'html':
            self.doc.write('\n.. toctree::\n')
            self.doc.write('  :maxdepth: 1\n')
            self.doc.write('  :titlesonly:\n\n')
        else:
            self.start_ul()

    def tocitem(self, item, file_name=None):
        if self.doc.target == 'man':
            self.li(item)
        elif file_name:
            self.doc.writeln('  %s' % file_name)
        else:
            self.doc.writeln('  %s' % item)

    def hidden_toctree(self):
        if self.doc.target == 'html':
            self.doc.write('\n.. toctree::\n')
            self.doc.write('  :maxdepth: 1\n')
            self.doc.write('  :hidden:\n\n')

    def hidden_tocitem(self, item):
        if self.doc.target == 'html':
            self.tocitem(item)

    def table_of_contents(self, title=None, depth=None):
        self.doc.write('.. contents:: ')
        if title is not None:
            self.doc.writeln(title)
        if depth is not None:
            self.doc.writeln('   :depth: %s' % depth)

    def start_sphinx_py_class(self, class_name):
        self.new_paragraph()
        self.doc.write('.. py:class:: %s' % class_name)
        self.indent()
        self.new_paragraph()

    def end_sphinx_py_class(self):
        self.dedent()
        self.new_paragraph()

    def start_sphinx_py_method(self, method_name, parameters=None):
        self.new_paragraph()
        content = '.. py:method:: %s' % method_name
        if parameters is not None:
            content += '(%s)' % parameters
        self.doc.write(content)
        self.indent()
        self.new_paragraph()

    def end_sphinx_py_method(self):
        self.dedent()
        self.new_paragraph()

    def start_sphinx_py_attr(self, attr_name):
        self.new_paragraph()
        self.doc.write('.. py:attribute:: %s' % attr_name)
        self.indent()
        self.new_paragraph()

    def end_sphinx_py_attr(self):
        self.dedent()
        self.new_paragraph()

    def write_py_doc_string(self, docstring):
        docstring_lines = docstring.splitlines()
        for docstring_line in docstring_lines:
            self.doc.writeln(docstring_line)

    def external_link(self, title, link):
        if self.doc.target == 'html':
            self.doc.write(f'`{title} <{link}>`_')
        else:
            self.doc.write(title)

    def internal_link(self, title, page):
        if self.doc.target == 'html':
            self.doc.write(f':doc:`{title} <{page}>`')
        else:
            self.doc.write(title)