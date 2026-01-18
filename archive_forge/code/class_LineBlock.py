import sys
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils.parsers.rst.roles import set_classes
from docutils.utils.code_analyzer import Lexer, LexerError, NumberLines
class LineBlock(Directive):
    option_spec = {'class': directives.class_option, 'name': directives.unchanged}
    has_content = True

    def run(self):
        self.assert_has_content()
        block = nodes.line_block(classes=self.options.get('class', []))
        self.add_name(block)
        node_list = [block]
        for line_text in self.content:
            text_nodes, messages = self.state.inline_text(line_text.strip(), self.lineno + self.content_offset)
            line = nodes.line(line_text, '', *text_nodes)
            if line_text.strip():
                line.indent = len(line_text) - len(line_text.lstrip())
            block += line
            node_list.extend(messages)
            self.content_offset += 1
        self.state.nest_line_block_lines(block)
        return node_list