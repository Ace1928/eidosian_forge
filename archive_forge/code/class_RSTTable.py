import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
class RSTTable(Table):

    def run(self):
        if not self.content:
            warning = self.state_machine.reporter.warning('Content block expected for the "%s" directive; none found.' % self.name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            return [warning]
        title, messages = self.make_title()
        node = nodes.Element()
        self.state.nested_parse(self.content, self.content_offset, node)
        if len(node) != 1 or not isinstance(node[0], nodes.table):
            error = self.state_machine.reporter.error('Error parsing content block for the "%s" directive: exactly one table expected.' % self.name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            return [error]
        table_node = node[0]
        table_node['classes'] += self.options.get('class', [])
        self.set_table_width(table_node)
        if 'align' in self.options:
            table_node['align'] = self.options.get('align')
        tgroup = table_node[0]
        if type(self.widths) == list:
            colspecs = [child for child in tgroup.children if child.tagname == 'colspec']
            for colspec, col_width in zip(colspecs, self.widths):
                colspec['colwidth'] = col_width
        if self.widths == 'auto':
            table_node['classes'] += ['colwidths-auto']
        elif self.widths:
            table_node['classes'] += ['colwidths-given']
        self.add_name(table_node)
        if title:
            table_node.insert(0, title)
        return [table_node] + messages