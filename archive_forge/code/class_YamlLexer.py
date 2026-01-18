import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, LexerContext, \
from pygments.token import Text, Comment, Keyword, Name, String, Number, \
class YamlLexer(ExtendedRegexLexer):
    """
    Lexer for `YAML <http://yaml.org/>`_, a human-friendly data serialization
    language.

    .. versionadded:: 0.11
    """
    name = 'YAML'
    aliases = ['yaml']
    filenames = ['*.yaml', '*.yml']
    mimetypes = ['text/x-yaml']

    def something(token_class):
        """Do not produce empty tokens."""

        def callback(lexer, match, context):
            text = match.group()
            if not text:
                return
            yield (match.start(), token_class, text)
            context.pos = match.end()
        return callback

    def reset_indent(token_class):
        """Reset the indentation levels."""

        def callback(lexer, match, context):
            text = match.group()
            context.indent_stack = []
            context.indent = -1
            context.next_indent = 0
            context.block_scalar_indent = None
            yield (match.start(), token_class, text)
            context.pos = match.end()
        return callback

    def save_indent(token_class, start=False):
        """Save a possible indentation level."""

        def callback(lexer, match, context):
            text = match.group()
            extra = ''
            if start:
                context.next_indent = len(text)
                if context.next_indent < context.indent:
                    while context.next_indent < context.indent:
                        context.indent = context.indent_stack.pop()
                    if context.next_indent > context.indent:
                        extra = text[context.indent:]
                        text = text[:context.indent]
            else:
                context.next_indent += len(text)
            if text:
                yield (match.start(), token_class, text)
            if extra:
                yield (match.start() + len(text), token_class.Error, extra)
            context.pos = match.end()
        return callback

    def set_indent(token_class, implicit=False):
        """Set the previously saved indentation level."""

        def callback(lexer, match, context):
            text = match.group()
            if context.indent < context.next_indent:
                context.indent_stack.append(context.indent)
                context.indent = context.next_indent
            if not implicit:
                context.next_indent += len(text)
            yield (match.start(), token_class, text)
            context.pos = match.end()
        return callback

    def set_block_scalar_indent(token_class):
        """Set an explicit indentation level for a block scalar."""

        def callback(lexer, match, context):
            text = match.group()
            context.block_scalar_indent = None
            if not text:
                return
            increment = match.group(1)
            if increment:
                current_indent = max(context.indent, 0)
                increment = int(increment)
                context.block_scalar_indent = current_indent + increment
            if text:
                yield (match.start(), token_class, text)
                context.pos = match.end()
        return callback

    def parse_block_scalar_empty_line(indent_token_class, content_token_class):
        """Process an empty line in a block scalar."""

        def callback(lexer, match, context):
            text = match.group()
            if context.block_scalar_indent is None or len(text) <= context.block_scalar_indent:
                if text:
                    yield (match.start(), indent_token_class, text)
            else:
                indentation = text[:context.block_scalar_indent]
                content = text[context.block_scalar_indent:]
                yield (match.start(), indent_token_class, indentation)
                yield (match.start() + context.block_scalar_indent, content_token_class, content)
            context.pos = match.end()
        return callback

    def parse_block_scalar_indent(token_class):
        """Process indentation spaces in a block scalar."""

        def callback(lexer, match, context):
            text = match.group()
            if context.block_scalar_indent is None:
                if len(text) <= max(context.indent, 0):
                    context.stack.pop()
                    context.stack.pop()
                    return
                context.block_scalar_indent = len(text)
            elif len(text) < context.block_scalar_indent:
                context.stack.pop()
                context.stack.pop()
                return
            if text:
                yield (match.start(), token_class, text)
                context.pos = match.end()
        return callback

    def parse_plain_scalar_indent(token_class):
        """Process indentation spaces in a plain scalar."""

        def callback(lexer, match, context):
            text = match.group()
            if len(text) <= context.indent:
                context.stack.pop()
                context.stack.pop()
                return
            if text:
                yield (match.start(), token_class, text)
                context.pos = match.end()
        return callback
    tokens = {'root': [('[ ]+(?=#|$)', Text), ('\\n+', Text), ('#[^\\n]*', Comment.Single), ('^%YAML(?=[ ]|$)', reset_indent(Name.Tag), 'yaml-directive'), ('^%TAG(?=[ ]|$)', reset_indent(Name.Tag), 'tag-directive'), ('^(?:---|\\.\\.\\.)(?=[ ]|$)', reset_indent(Name.Namespace), 'block-line'), ('[ ]*(?!\\s|$)', save_indent(Text, start=True), ('block-line', 'indentation'))], 'ignored-line': [('[ ]+(?=#|$)', Text), ('#[^\\n]*', Comment.Single), ('\\n', Text, '#pop:2')], 'yaml-directive': [('([ ]+)([0-9]+\\.[0-9]+)', bygroups(Text, Number), 'ignored-line')], 'tag-directive': [("([ ]+)(!|![\\w-]*!)([ ]+)(!|!?[\\w;/?:@&=+$,.!~*\\'()\\[\\]%-]+)", bygroups(Text, Keyword.Type, Text, Keyword.Type), 'ignored-line')], 'indentation': [('[ ]*$', something(Text), '#pop:2'), ('[ ]+(?=[?:-](?:[ ]|$))', save_indent(Text)), ('[?:-](?=[ ]|$)', set_indent(Punctuation.Indicator)), ('[ ]*', save_indent(Text), '#pop')], 'block-line': [('[ ]*(?=#|$)', something(Text), '#pop'), ('[ ]+', Text), include('descriptors'), include('block-nodes'), include('flow-nodes'), ('(?=[^\\s?:,\\[\\]{}#&*!|>\\\'"%@`-]|[?:-]\\S)', something(Name.Variable), 'plain-scalar-in-block-context')], 'descriptors': [("!<[\\w#;/?:@&=+$,.!~*\\'()\\[\\]%-]+>", Keyword.Type), ("!(?:[\\w-]+!)?[\\w#;/?:@&=+$,.!~*\\'()\\[\\]%-]+", Keyword.Type), ('&[\\w-]+', Name.Label), ('\\*[\\w-]+', Name.Variable)], 'block-nodes': [(':(?=[ ]|$)', set_indent(Punctuation.Indicator, implicit=True)), ('[|>]', Punctuation.Indicator, ('block-scalar-content', 'block-scalar-header'))], 'flow-nodes': [('\\[', Punctuation.Indicator, 'flow-sequence'), ('\\{', Punctuation.Indicator, 'flow-mapping'), ("\\'", String, 'single-quoted-scalar'), ('\\"', String, 'double-quoted-scalar')], 'flow-collection': [('[ ]+', Text), ('\\n+', Text), ('#[^\\n]*', Comment.Single), ('[?:,]', Punctuation.Indicator), include('descriptors'), include('flow-nodes'), ('(?=[^\\s?:,\\[\\]{}#&*!|>\\\'"%@`])', something(Name.Variable), 'plain-scalar-in-flow-context')], 'flow-sequence': [include('flow-collection'), ('\\]', Punctuation.Indicator, '#pop')], 'flow-mapping': [include('flow-collection'), ('\\}', Punctuation.Indicator, '#pop')], 'block-scalar-content': [('\\n', Text), ('^[ ]+$', parse_block_scalar_empty_line(Text, Name.Constant)), ('^[ ]*', parse_block_scalar_indent(Text)), ('[\\S\\t ]+', Name.Constant)], 'block-scalar-header': [('([1-9])?[+-]?(?=[ ]|$)', set_block_scalar_indent(Punctuation.Indicator), 'ignored-line'), ('[+-]?([1-9])?(?=[ ]|$)', set_block_scalar_indent(Punctuation.Indicator), 'ignored-line')], 'quoted-scalar-whitespaces': [('^[ ]+', Text), ('[ ]+$', Text), ('\\n+', Text), ('[ ]+', Name.Variable)], 'single-quoted-scalar': [include('quoted-scalar-whitespaces'), ("\\'\\'", String.Escape), ("[^\\s\\']+", String), ("\\'", String, '#pop')], 'double-quoted-scalar': [include('quoted-scalar-whitespaces'), ('\\\\[0abt\\tn\\nvfre "\\\\N_LP]', String), ('\\\\(?:x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})', String.Escape), ('[^\\s"\\\\]+', String), ('"', String, '#pop')], 'plain-scalar-in-block-context-new-line': [('^[ ]+$', Text), ('\\n+', Text), ('^(?=---|\\.\\.\\.)', something(Name.Namespace), '#pop:3'), ('^[ ]*', parse_plain_scalar_indent(Text), '#pop')], 'plain-scalar-in-block-context': [('[ ]*(?=:[ ]|:$)', something(Text), '#pop'), ('[ ]+(?=#)', Text, '#pop'), ('[ ]+$', Text), ('\\n+', Text, 'plain-scalar-in-block-context-new-line'), ('[ ]+', Literal.Scalar.Plain), ('(?::(?!\\s)|[^\\s:])+', Literal.Scalar.Plain)], 'plain-scalar-in-flow-context': [('[ ]*(?=[,:?\\[\\]{}])', something(Text), '#pop'), ('[ ]+(?=#)', Text, '#pop'), ('^[ ]+', Text), ('[ ]+$', Text), ('\\n+', Text), ('[ ]+', Name.Variable), ('[^\\s,:?\\[\\]{}]+', Name.Variable)]}

    def get_tokens_unprocessed(self, text=None, context=None):
        if context is None:
            context = YamlLexerContext(text, 0)
        return super(YamlLexer, self).get_tokens_unprocessed(text, context)