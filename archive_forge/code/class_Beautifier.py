import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
class Beautifier:

    def __init__(self, opts=None):
        import jsbeautifier.javascript.acorn as acorn
        self.acorn = acorn
        self._options = BeautifierOptions(opts)
        self._blank_state()

    def _blank_state(self, js_source_text=None):
        if js_source_text is None:
            js_source_text = ''
        self._flags = None
        self._previous_flags = None
        self._flag_store = []
        self._tokens = None
        if self._options.eol == 'auto':
            self._options.eol = '\n'
            if self.acorn.lineBreak.search(js_source_text or ''):
                self._options.eol = self.acorn.lineBreak.search(js_source_text).group()
        baseIndentString = re.search('^[\t ]*', js_source_text).group(0)
        self._last_last_text = ''
        self._output = Output(self._options, baseIndentString)
        self._output.raw = self._options.test_output_raw
        self.set_mode(MODE.BlockStatement)
        return js_source_text

    def beautify(self, source_text='', opts=None):
        if opts is not None:
            self._options = BeautifierOptions(opts)
        source_text = source_text or ''
        if self._options.disabled:
            return source_text
        source_text = self._blank_state(source_text)
        source_text = self.unpack(source_text, self._options.eval_code)
        self._tokens = Tokenizer(source_text, self._options).tokenize()
        for current_token in self._tokens:
            self.handle_token(current_token)
            self._last_last_text = self._flags.last_token.text
            self._flags.last_token = current_token
        sweet_code = self._output.get_code(self._options.eol)
        return sweet_code

    def handle_token(self, current_token, preserve_statement_flags=False):
        if current_token.type == TOKEN.START_EXPR:
            self.handle_start_expr(current_token)
        elif current_token.type == TOKEN.END_EXPR:
            self.handle_end_expr(current_token)
        elif current_token.type == TOKEN.START_BLOCK:
            self.handle_start_block(current_token)
        elif current_token.type == TOKEN.END_BLOCK:
            self.handle_end_block(current_token)
        elif current_token.type == TOKEN.WORD:
            self.handle_word(current_token)
        elif current_token.type == TOKEN.RESERVED:
            self.handle_word(current_token)
        elif current_token.type == TOKEN.SEMICOLON:
            self.handle_semicolon(current_token)
        elif current_token.type == TOKEN.STRING:
            self.handle_string(current_token)
        elif current_token.type == TOKEN.EQUALS:
            self.handle_equals(current_token)
        elif current_token.type == TOKEN.OPERATOR:
            self.handle_operator(current_token)
        elif current_token.type == TOKEN.COMMA:
            self.handle_comma(current_token)
        elif current_token.type == TOKEN.BLOCK_COMMENT:
            self.handle_block_comment(current_token, preserve_statement_flags)
        elif current_token.type == TOKEN.COMMENT:
            self.handle_comment(current_token, preserve_statement_flags)
        elif current_token.type == TOKEN.DOT:
            self.handle_dot(current_token)
        elif current_token.type == TOKEN.EOF:
            self.handle_eof(current_token)
        elif current_token.type == TOKEN.UNKNOWN:
            self.handle_unknown(current_token, preserve_statement_flags)
        else:
            self.handle_unknown(current_token, preserve_statement_flags)

    def handle_whitespace_and_comments(self, current_token, preserve_statement_flags=False):
        newlines = current_token.newlines
        keep_whitespace = self._options.keep_array_indentation and self.is_array(self._flags.mode)
        if current_token.comments_before is not None:
            for comment_token in current_token.comments_before:
                self.handle_whitespace_and_comments(comment_token, preserve_statement_flags)
                self.handle_token(comment_token, preserve_statement_flags)
        if keep_whitespace:
            for i in range(newlines):
                self.print_newline(i > 0, preserve_statement_flags)
        else:
            if self._options.max_preserve_newlines != 0 and newlines > self._options.max_preserve_newlines:
                newlines = self._options.max_preserve_newlines
            if self._options.preserve_newlines and newlines > 1:
                self.print_newline(False, preserve_statement_flags)
                for i in range(1, newlines):
                    self.print_newline(True, preserve_statement_flags)

    def unpack(self, source, evalcode=False):
        import jsbeautifier.unpackers as unpackers
        try:
            return unpackers.run(source, evalcode)
        except unpackers.UnpackingError:
            return source

    def is_array(self, mode):
        return mode == MODE.ArrayLiteral

    def is_expression(self, mode):
        return mode == MODE.Expression or mode == MODE.ForInitializer or mode == MODE.Conditional
    _newline_restricted_tokens = frozenset(['async', 'break', 'continue', 'return', 'throw', 'yield'])

    def allow_wrap_or_preserved_newline(self, current_token, force_linewrap=False):
        if self._output.just_added_newline():
            return
        shouldPreserveOrForce = self._options.preserve_newlines and current_token.newlines or force_linewrap
        operatorLogicApplies = self._flags.last_token.text in Tokenizer.positionable_operators or current_token.text in Tokenizer.positionable_operators
        if operatorLogicApplies:
            shouldPrintOperatorNewline = self._flags.last_token.text in Tokenizer.positionable_operators and self._options.operator_position in OPERATOR_POSITION_BEFORE_OR_PRESERVE or current_token.text in Tokenizer.positionable_operators
            shouldPreserveOrForce = shouldPreserveOrForce and shouldPrintOperatorNewline
        if shouldPreserveOrForce:
            self.print_newline(preserve_statement_flags=True)
        elif self._options.wrap_line_length > 0:
            if reserved_array(self._flags.last_token, self._newline_restricted_tokens):
                return
            self._output.set_wrap_point()

    def print_newline(self, force_newline=False, preserve_statement_flags=False):
        if not preserve_statement_flags:
            if self._flags.last_token.text != ';' and self._flags.last_token.text != ',' and (self._flags.last_token.text != '=') and (self._flags.last_token.type != TOKEN.OPERATOR or self._flags.last_token.text == '--' or self._flags.last_token.text == '++'):
                next_token = self._tokens.peek()
                while self._flags.mode == MODE.Statement and (not (self._flags.if_block and reserved_word(next_token, 'else'))) and (not self._flags.do_block):
                    self.restore_mode()
        if self._output.add_new_line(force_newline):
            self._flags.multiline_frame = True

    def print_token_line_indentation(self, current_token):
        if self._output.just_added_newline():
            line = self._output.current_line
            if self._options.keep_array_indentation and current_token.newlines and (self.is_array(self._flags.mode) or current_token.text == '['):
                line.set_indent(-1)
                line.push(current_token.whitespace_before)
                self._output.space_before_token = False
            elif self._output.set_indent(self._flags.indentation_level, self._flags.alignment):
                self._flags.line_indent_level = self._flags.indentation_level

    def print_token(self, current_token, s=None):
        if self._output.raw:
            self._output.add_raw_token(current_token)
            return
        if self._options.comma_first and current_token.previous and (current_token.previous.type == TOKEN.COMMA) and self._output.just_added_newline():
            if self._output.previous_line.last() == ',':
                popped = self._output.previous_line.pop()
                if self._output.previous_line.is_empty():
                    self._output.previous_line.push(popped)
                    self._output.trim(True)
                    self._output.current_line.pop()
                    self._output.trim()
                self.print_token_line_indentation(current_token)
                self._output.add_token(',')
                self._output.space_before_token = True
        if s is None:
            s = current_token.text
        self.print_token_line_indentation(current_token)
        self._output.non_breaking_space = True
        self._output.add_token(s)
        if self._output.previous_token_wrapped:
            self._flags.multiline_frame = True

    def indent(self):
        self._flags.indentation_level += 1
        self._output.set_indent(self._flags.indentation_level, self._flags.alignment)

    def deindent(self):
        allow_deindent = self._flags.indentation_level > 0 and (self._flags.parent is None or self._flags.indentation_level > self._flags.parent.indentation_level)
        if allow_deindent:
            self._flags.indentation_level -= 1
        self._output.set_indent(self._flags.indentation_level, self._flags.alignment)

    def set_mode(self, mode):
        if self._flags:
            self._flag_store.append(self._flags)
            self._previous_flags = self._flags
        else:
            self._previous_flags = BeautifierFlags(mode)
        self._flags = BeautifierFlags(mode)
        self._flags.apply_base(self._previous_flags, self._output.just_added_newline())
        self._flags.start_line_index = self._output.get_line_number()
        self._output.set_indent(self._flags.indentation_level, self._flags.alignment)

    def restore_mode(self):
        if len(self._flag_store) > 0:
            self._previous_flags = self._flags
            self._flags = self._flag_store.pop()
            if self._previous_flags.mode == MODE.Statement:
                remove_redundant_indentation(self._output, self._previous_flags)
        self._output.set_indent(self._flags.indentation_level, self._flags.alignment)

    def start_of_object_property(self):
        return self._flags.parent.mode == MODE.ObjectLiteral and self._flags.mode == MODE.Statement and (self._flags.last_token.text == ':' and self._flags.ternary_depth == 0 or reserved_array(self._flags.last_token, ['get', 'set']))

    def start_of_statement(self, current_token):
        start = False
        start = start or (reserved_array(self._flags.last_token, ['var', 'let', 'const']) and current_token.type == TOKEN.WORD)
        start = start or reserved_word(self._flags.last_token, 'do')
        start = start or (not (self._flags.parent.mode == MODE.ObjectLiteral and self._flags.mode == MODE.Statement) and reserved_array(self._flags.last_token, self._newline_restricted_tokens) and (not current_token.newlines))
        start = start or (reserved_word(self._flags.last_token, 'else') and (not (reserved_word(current_token, 'if') and current_token.comments_before is None)))
        start = start or (self._flags.last_token.type == TOKEN.END_EXPR and (self._previous_flags.mode == MODE.ForInitializer or self._previous_flags.mode == MODE.Conditional))
        start = start or (self._flags.last_token.type == TOKEN.WORD and self._flags.mode == MODE.BlockStatement and (not self._flags.in_case) and (not (current_token.text == '--' or current_token.text == '++')) and (self._last_last_text != 'function') and (current_token.type != TOKEN.WORD) and (current_token.type != TOKEN.RESERVED))
        start = start or (self._flags.mode == MODE.ObjectLiteral and (self._flags.last_token.text == ':' and self._flags.ternary_depth == 0 or reserved_array(self._flags.last_token, ['get', 'set'])))
        if start:
            self.set_mode(MODE.Statement)
            self.indent()
            self.handle_whitespace_and_comments(current_token, True)
            if not self.start_of_object_property():
                self.allow_wrap_or_preserved_newline(current_token, reserved_array(current_token, ['do', 'for', 'if', 'while']))
            return True
        else:
            return False

    def handle_start_expr(self, current_token):
        if self.start_of_statement(current_token):
            pass
        else:
            self.handle_whitespace_and_comments(current_token)
        next_mode = MODE.Expression
        if current_token.text == '[':
            if self._flags.last_token.type == TOKEN.WORD or self._flags.last_token.text == ')':
                if reserved_array(self._flags.last_token, Tokenizer.line_starters):
                    self._output.space_before_token = True
                self.print_token(current_token)
                self.set_mode(next_mode)
                self.indent()
                if self._options.space_in_paren:
                    self._output.space_before_token = True
                return
            next_mode = MODE.ArrayLiteral
            if self.is_array(self._flags.mode):
                if self._flags.last_token.text == '[' or (self._flags.last_token.text == ',' and (self._last_last_text == ']' or self._last_last_text == '}')):
                    if not self._options.keep_array_indentation:
                        self.print_newline()
            if self._flags.last_token.type not in [TOKEN.START_EXPR, TOKEN.END_EXPR, TOKEN.WORD, TOKEN.OPERATOR, TOKEN.DOT]:
                self._output.space_before_token = True
        else:
            if self._flags.last_token.type == TOKEN.RESERVED:
                if self._flags.last_token.text == 'for':
                    self._output.space_before_token = self._options.space_before_conditional
                    next_mode = MODE.ForInitializer
                elif self._flags.last_token.text in ['if', 'while', 'switch']:
                    self._output.space_before_token = self._options.space_before_conditional
                    next_mode = MODE.Conditional
                elif self._flags.last_word in ['await', 'async']:
                    self._output.space_before_token = True
                elif self._flags.last_token.text == 'import' and current_token.whitespace_before == '':
                    self._output.space_before_token = False
                elif self._flags.last_token.text in Tokenizer.line_starters or self._flags.last_token.text == 'catch':
                    self._output.space_before_token = True
            elif self._flags.last_token.type in [TOKEN.EQUALS, TOKEN.OPERATOR]:
                if not self.start_of_object_property():
                    self.allow_wrap_or_preserved_newline(current_token)
            elif self._flags.last_token.type == TOKEN.WORD:
                self._output.space_before_token = False
                peek_back_two = self._tokens.peek(-3)
                if self._options.space_after_named_function and peek_back_two:
                    peek_back_three = self._tokens.peek(-4)
                    if reserved_array(peek_back_two, ['async', 'function']) or (peek_back_two.text == '*' and reserved_array(peek_back_three, ['async', 'function'])):
                        self._output.space_before_token = True
                    elif self._flags.mode == MODE.ObjectLiteral:
                        if (peek_back_two.text == '{' or peek_back_two.text == ',') or (peek_back_two.text == '*' and (peek_back_three.text == '{' or peek_back_three.text == ',')):
                            self._output.space_before_token = True
                    elif self._flags.parent and self._flags.parent.class_start_block:
                        self._output.space_before_token = True
            else:
                self.allow_wrap_or_preserved_newline(current_token)
            if self._flags.last_token.type == TOKEN.RESERVED and (self._flags.last_word == 'function' or self._flags.last_word == 'typeof') or (self._flags.last_token.text == '*' and (self._last_last_text in ['function', 'yield'] or (self._flags.mode == MODE.ObjectLiteral and self._last_last_text in ['{', ',']))):
                self._output.space_before_token = self._options.space_after_anon_function
        if self._flags.last_token.text == ';' or self._flags.last_token.type == TOKEN.START_BLOCK:
            self.print_newline()
        elif self._flags.last_token.type in [TOKEN.END_EXPR, TOKEN.START_EXPR, TOKEN.END_BLOCK, TOKEN.COMMA] or self._flags.last_token.text == '.':
            self.allow_wrap_or_preserved_newline(current_token, current_token.newlines)
        self.print_token(current_token)
        self.set_mode(next_mode)
        if self._options.space_in_paren:
            self._output.space_before_token = True
        self.indent()

    def handle_end_expr(self, current_token):
        while self._flags.mode == MODE.Statement:
            self.restore_mode()
        self.handle_whitespace_and_comments(current_token)
        if self._flags.multiline_frame:
            self.allow_wrap_or_preserved_newline(current_token, current_token.text == ']' and self.is_array(self._flags.mode) and (not self._options.keep_array_indentation))
        if self._options.space_in_paren:
            if self._flags.last_token.type == TOKEN.START_EXPR and (not self._options.space_in_empty_paren):
                self._output.space_before_token = False
                self._output.trim()
            else:
                self._output.space_before_token = True
        self.deindent()
        self.print_token(current_token)
        self.restore_mode()
        remove_redundant_indentation(self._output, self._previous_flags)
        if self._flags.do_while and self._previous_flags.mode == MODE.Conditional:
            self._previous_flags.mode = MODE.Expression
            self._flags.do_block = False
            self._flags.do_while = False

    def handle_start_block(self, current_token):
        self.handle_whitespace_and_comments(current_token)
        next_token = self._tokens.peek()
        second_token = self._tokens.peek(1)
        if self._flags.last_word == 'switch' and self._flags.last_token.type == TOKEN.END_EXPR:
            self.set_mode(MODE.BlockStatement)
            self._flags.in_case_statement = True
        elif self._flags.case_body:
            self.set_mode(MODE.BlockStatement)
        elif second_token is not None and (second_token.text in [':', ','] and next_token.type in [TOKEN.STRING, TOKEN.WORD, TOKEN.RESERVED] or (next_token.text in ['get', 'set', '...'] and second_token.type in [TOKEN.WORD, TOKEN.RESERVED])):
            if self._last_last_text in ['class', 'interface'] and second_token.text not in [':', ',']:
                self.set_mode(MODE.BlockStatement)
            else:
                self.set_mode(MODE.ObjectLiteral)
        elif self._flags.last_token.type == TOKEN.OPERATOR and self._flags.last_token.text == '=>':
            self.set_mode(MODE.BlockStatement)
        elif self._flags.last_token.type in [TOKEN.EQUALS, TOKEN.START_EXPR, TOKEN.COMMA, TOKEN.OPERATOR] or reserved_array(self._flags.last_token, ['return', 'throw', 'import', 'default']):
            self.set_mode(MODE.ObjectLiteral)
        else:
            self.set_mode(MODE.BlockStatement)
        if self._flags.last_token:
            if reserved_array(self._flags.last_token.previous, ['class', 'extends']):
                self._flags.class_start_block = True
        empty_braces = next_token is not None and next_token.comments_before is None and (next_token.text == '}')
        empty_anonymous_function = empty_braces and self._flags.last_word == 'function' and (self._flags.last_token.type == TOKEN.END_EXPR)
        if self._options.brace_preserve_inline:
            index = 0
            check_token = None
            self._flags.inline_frame = True
            do_loop = True
            while do_loop:
                index += 1
                check_token = self._tokens.peek(index - 1)
                if check_token.newlines:
                    self._flags.inline_frame = False
                do_loop = check_token.type != TOKEN.EOF and (not (check_token.type == TOKEN.END_BLOCK and check_token.opened == current_token))
        if (self._options.brace_style == 'expand' or (self._options.brace_style == 'none' and current_token.newlines)) and (not self._flags.inline_frame):
            if self._flags.last_token.type != TOKEN.OPERATOR and (empty_anonymous_function or self._flags.last_token.type == TOKEN.EQUALS or (reserved_array(self._flags.last_token, _special_word_set) and self._flags.last_token.text != 'else')):
                self._output.space_before_token = True
            else:
                self.print_newline(preserve_statement_flags=True)
        elif self.is_array(self._previous_flags.mode) and (self._flags.last_token.type == TOKEN.START_EXPR or self._flags.last_token.type == TOKEN.COMMA):
            if self._flags.inline_frame:
                self.allow_wrap_or_preserved_newline(current_token)
                self._flags.inline_frame = True
                self._previous_flags.multiline_frame = self._previous_flags.multiline_frame or self._flags.multiline_frame
                self._flags.multiline_frame = False
            elif self._flags.last_token.type == TOKEN.COMMA:
                self._output.space_before_token = True
        elif self._flags.last_token.type not in [TOKEN.OPERATOR, TOKEN.START_EXPR]:
            if self._flags.last_token.type in [TOKEN.START_BLOCK, TOKEN.SEMICOLON] and (not self._flags.inline_frame):
                self.print_newline()
            else:
                self._output.space_before_token = True
        self.print_token(current_token)
        self.indent()
        if not empty_braces and (not (self._options.brace_preserve_inline and self._flags.inline_frame)):
            self.print_newline()

    def handle_end_block(self, current_token):
        self.handle_whitespace_and_comments(current_token)
        while self._flags.mode == MODE.Statement:
            self.restore_mode()
        empty_braces = self._flags.last_token.type == TOKEN.START_BLOCK
        if self._flags.inline_frame and (not empty_braces):
            self._output.space_before_token = True
        elif self._options.brace_style == 'expand':
            if not empty_braces:
                self.print_newline()
        elif not empty_braces:
            if self.is_array(self._flags.mode) and self._options.keep_array_indentation:
                self._options.keep_array_indentation = False
                self.print_newline()
                self._options.keep_array_indentation = True
            else:
                self.print_newline()
        self.restore_mode()
        self.print_token(current_token)

    def handle_word(self, current_token):
        if current_token.type == TOKEN.RESERVED:
            if current_token.text in ['set', 'get'] and self._flags.mode != MODE.ObjectLiteral:
                current_token.type = TOKEN.WORD
            elif current_token.text == 'import' and self._tokens.peek().text in ['(', '.']:
                current_token.type = TOKEN.WORD
            elif current_token.text in ['as', 'from'] and (not self._flags.import_block):
                current_token.type = TOKEN.WORD
            elif self._flags.mode == MODE.ObjectLiteral:
                next_token = self._tokens.peek()
                if next_token.text == ':':
                    current_token.type = TOKEN.WORD
        if self.start_of_statement(current_token):
            if reserved_array(self._flags.last_token, ['var', 'let', 'const']) and current_token.type == TOKEN.WORD:
                self._flags.declaration_statement = True
        elif current_token.newlines and (not self.is_expression(self._flags.mode)) and (self._flags.last_token.type != TOKEN.OPERATOR or (self._flags.last_token.text == '--' or self._flags.last_token.text == '++')) and (self._flags.last_token.type != TOKEN.EQUALS) and (self._options.preserve_newlines or not reserved_array(self._flags.last_token, ['var', 'let', 'const', 'set', 'get'])):
            self.handle_whitespace_and_comments(current_token)
            self.print_newline()
        else:
            self.handle_whitespace_and_comments(current_token)
        if self._flags.do_block and (not self._flags.do_while):
            if reserved_word(current_token, 'while'):
                self._output.space_before_token = True
                self.print_token(current_token)
                self._output.space_before_token = True
                self._flags.do_while = True
                return
            else:
                self.print_newline()
                self._flags.do_block = False
        if self._flags.if_block:
            if not self._flags.else_block and reserved_word(current_token, 'else'):
                self._flags.else_block = True
            else:
                while self._flags.mode == MODE.Statement:
                    self.restore_mode()
                self._flags.if_block = False
        if self._flags.in_case_statement and reserved_array(current_token, ['case', 'default']):
            self.print_newline()
            if not self._flags.case_block and (self._flags.case_body or self._options.jslint_happy):
                self.deindent()
            self._flags.case_body = False
            self.print_token(current_token)
            self._flags.in_case = True
            return
        if self._flags.last_token.type in [TOKEN.COMMA, TOKEN.START_EXPR, TOKEN.EQUALS, TOKEN.OPERATOR]:
            if not self.start_of_object_property() and (not (self._flags.last_token.text in ['+', '-'] and self._last_last_text == ':' and (self._flags.parent.mode == MODE.ObjectLiteral))):
                self.allow_wrap_or_preserved_newline(current_token)
        if reserved_word(current_token, 'function'):
            if self._flags.last_token.text in ['}', ';'] or (self._output.just_added_newline() and (not (self._flags.last_token.text in ['(', '[', '{', ':', '=', ','] or self._flags.last_token.type == TOKEN.OPERATOR))):
                if not self._output.just_added_blankline() and current_token.comments_before is None:
                    self.print_newline()
                    self.print_newline(True)
            if self._flags.last_token.type == TOKEN.RESERVED or self._flags.last_token.type == TOKEN.WORD:
                if reserved_array(self._flags.last_token, ['get', 'set', 'new', 'export']) or reserved_array(self._flags.last_token, self._newline_restricted_tokens):
                    self._output.space_before_token = True
                elif reserved_word(self._flags.last_token, 'default') and self._last_last_text == 'export':
                    self._output.space_before_token = True
                elif self._flags.last_token.text == 'declare':
                    self._output.space_before_token = True
                else:
                    self.print_newline()
            elif self._flags.last_token.type == TOKEN.OPERATOR or self._flags.last_token.text == '=':
                self._output.space_before_token = True
            elif not self._flags.multiline_frame and (self.is_expression(self._flags.mode) or self.is_array(self._flags.mode)):
                pass
            else:
                self.print_newline()
            self.print_token(current_token)
            self._flags.last_word = current_token.text
            return
        prefix = 'NONE'
        if self._flags.last_token.type == TOKEN.END_BLOCK:
            if self._previous_flags.inline_frame:
                prefix = 'SPACE'
            elif not reserved_array(current_token, ['else', 'catch', 'finally', 'from']):
                prefix = 'NEWLINE'
            elif self._options.brace_style in ['expand', 'end-expand'] or (self._options.brace_style == 'none' and current_token.newlines):
                prefix = 'NEWLINE'
            else:
                prefix = 'SPACE'
                self._output.space_before_token = True
        elif self._flags.last_token.type == TOKEN.SEMICOLON and self._flags.mode == MODE.BlockStatement:
            prefix = 'NEWLINE'
        elif self._flags.last_token.type == TOKEN.SEMICOLON and self.is_expression(self._flags.mode):
            prefix = 'SPACE'
        elif self._flags.last_token.type == TOKEN.STRING:
            prefix = 'NEWLINE'
        elif self._flags.last_token.type == TOKEN.RESERVED or self._flags.last_token.type == TOKEN.WORD or (self._flags.last_token.text == '*' and (self._last_last_text in ['function', 'yield'] or (self._flags.mode == MODE.ObjectLiteral and self._last_last_text in ['{', ',']))):
            prefix = 'SPACE'
        elif self._flags.last_token.type == TOKEN.START_BLOCK:
            if self._flags.inline_frame:
                prefix = 'SPACE'
            else:
                prefix = 'NEWLINE'
        elif self._flags.last_token.type == TOKEN.END_EXPR:
            self._output.space_before_token = True
            prefix = 'NEWLINE'
        if reserved_array(current_token, Tokenizer.line_starters) and self._flags.last_token.text != ')':
            if self._flags.inline_frame or self._flags.last_token.text == 'else ' or self._flags.last_token.text == 'export':
                prefix = 'SPACE'
            else:
                prefix = 'NEWLINE'
        if reserved_array(current_token, ['else', 'catch', 'finally']):
            if (not (self._flags.last_token.type == TOKEN.END_BLOCK and self._previous_flags.mode == MODE.BlockStatement) or self._options.brace_style == 'expand' or self._options.brace_style == 'end-expand' or (self._options.brace_style == 'none' and current_token.newlines)) and (not self._flags.inline_frame):
                self.print_newline()
            else:
                self._output.trim(True)
                if self._output.current_line.last() != '}':
                    self.print_newline()
                self._output.space_before_token = True
        elif prefix == 'NEWLINE':
            if reserved_array(self._flags.last_token, _special_word_set):
                self._output.space_before_token = True
            elif self._flags.last_token.text == 'declare' and reserved_array(current_token, ['var', 'let', 'const']):
                self._output.space_before_token = True
            elif self._flags.last_token.type != TOKEN.END_EXPR:
                if (self._flags.last_token.type != TOKEN.START_EXPR or not reserved_array(current_token, ['var', 'let', 'const'])) and self._flags.last_token.text != ':':
                    if reserved_word(current_token, 'if') and self._flags.last_token.text == 'else':
                        self._output.space_before_token = True
                    else:
                        self.print_newline()
            elif reserved_array(current_token, Tokenizer.line_starters) and self._flags.last_token.text != ')':
                self.print_newline()
        elif self._flags.multiline_frame and self.is_array(self._flags.mode) and (self._flags.last_token.text == ',') and (self._last_last_text == '}'):
            self.print_newline()
        elif prefix == 'SPACE':
            self._output.space_before_token = True
        if current_token.previous and (current_token.previous.type == TOKEN.WORD or current_token.previous.type == TOKEN.RESERVED):
            self._output.space_before_token = True
        self.print_token(current_token)
        self._flags.last_word = current_token.text
        if current_token.type == TOKEN.RESERVED:
            if current_token.text == 'do':
                self._flags.do_block = True
            elif current_token.text == 'if':
                self._flags.if_block = True
            elif current_token.text == 'import':
                self._flags.import_block = True
            elif current_token.text == 'from' and self._flags.import_block:
                self._flags.import_block = False

    def handle_semicolon(self, current_token):
        if self.start_of_statement(current_token):
            self._output.space_before_token = False
        else:
            self.handle_whitespace_and_comments(current_token)
        next_token = self._tokens.peek()
        while self._flags.mode == MODE.Statement and (not (self._flags.if_block and reserved_word(next_token, 'else'))) and (not self._flags.do_block):
            self.restore_mode()
        if self._flags.import_block:
            self._flags.import_block = False
        self.print_token(current_token)

    def handle_string(self, current_token):
        if current_token.text[0] == '`' and current_token.newlines == 0 and (current_token.whitespace_before == '') and (self._flags.last_token.type == TOKEN.WORD or current_token.previous.text == ')'):
            pass
        elif self.start_of_statement(current_token):
            self._output.space_before_token = True
        else:
            self.handle_whitespace_and_comments(current_token)
            if self._flags.last_token.type in [TOKEN.RESERVED, TOKEN.WORD] or self._flags.inline_frame:
                self._output.space_before_token = True
            elif self._flags.last_token.type in [TOKEN.COMMA, TOKEN.START_EXPR, TOKEN.EQUALS, TOKEN.OPERATOR]:
                if not self.start_of_object_property():
                    self.allow_wrap_or_preserved_newline(current_token)
            elif current_token.text[0] == '`' and self._flags.last_token.type == TOKEN.END_EXPR and (current_token.previous.text in [']', ')']) and (current_token.newlines == 0):
                self._output.space_before_token = True
            else:
                self.print_newline()
        self.print_token(current_token)

    def handle_equals(self, current_token):
        if self.start_of_statement(current_token):
            pass
        else:
            self.handle_whitespace_and_comments(current_token)
        if self._flags.declaration_statement:
            self._flags.declaration_assignment = True
        self._output.space_before_token = True
        self.print_token(current_token)
        self._output.space_before_token = True

    def handle_comma(self, current_token):
        self.handle_whitespace_and_comments(current_token, True)
        self.print_token(current_token)
        self._output.space_before_token = True
        if self._flags.declaration_statement:
            if self.is_expression(self._flags.parent.mode):
                self._flags.declaration_assignment = False
            if self._flags.declaration_assignment:
                self._flags.declaration_assignment = False
                self.print_newline(preserve_statement_flags=True)
            elif self._options.comma_first:
                self.allow_wrap_or_preserved_newline(current_token)
        elif self._flags.mode == MODE.ObjectLiteral or (self._flags.mode == MODE.Statement and self._flags.parent.mode == MODE.ObjectLiteral):
            if self._flags.mode == MODE.Statement:
                self.restore_mode()
            if not self._flags.inline_frame:
                self.print_newline()
        elif self._options.comma_first:
            self.allow_wrap_or_preserved_newline(current_token)

    def handle_operator(self, current_token):
        isGeneratorAsterisk = current_token.text == '*' and (reserved_array(self._flags.last_token, ['function', 'yield']) or self._flags.last_token.type in [TOKEN.START_BLOCK, TOKEN.COMMA, TOKEN.END_BLOCK, TOKEN.SEMICOLON])
        isUnary = current_token.text in ['+', '-'] and (self._flags.last_token.type in [TOKEN.START_BLOCK, TOKEN.START_EXPR, TOKEN.EQUALS, TOKEN.OPERATOR] or self._flags.last_token.text in Tokenizer.line_starters or self._flags.last_token.text == ',')
        if self.start_of_statement(current_token):
            pass
        else:
            preserve_statement_flags = not isGeneratorAsterisk
            self.handle_whitespace_and_comments(current_token, preserve_statement_flags)
        if current_token.text == '*' and self._flags.last_token.type == TOKEN.DOT:
            self.print_token(current_token)
            return
        if current_token.text == '::':
            self.print_token(current_token)
            return
        if current_token.text in ['-', '+'] and self.start_of_object_property():
            self.print_token(current_token)
            return
        if self._flags.last_token.type == TOKEN.OPERATOR and self._options.operator_position in OPERATOR_POSITION_BEFORE_OR_PRESERVE:
            self.allow_wrap_or_preserved_newline(current_token)
        if current_token.text == ':' and self._flags.in_case:
            self.print_token(current_token)
            self._flags.in_case = False
            self._flags.case_body = True
            if self._tokens.peek().type != TOKEN.START_BLOCK:
                self.indent()
                self.print_newline()
                self._flags.case_block = False
            else:
                self._output.space_before_token = True
                self._flags.case_block = True
            return
        space_before = True
        space_after = True
        in_ternary = False
        if current_token.text == ':':
            if self._flags.ternary_depth == 0:
                space_before = False
            else:
                self._flags.ternary_depth -= 1
                in_ternary = True
        elif current_token.text == '?':
            self._flags.ternary_depth += 1
        if not isUnary and (not isGeneratorAsterisk) and self._options.preserve_newlines and (current_token.text in Tokenizer.positionable_operators):
            isColon = current_token.text == ':'
            isTernaryColon = isColon and in_ternary
            isOtherColon = isColon and (not in_ternary)
            if self._options.operator_position == OPERATOR_POSITION['before_newline']:
                self._output.space_before_token = not isOtherColon
                self.print_token(current_token)
                if not isColon or isTernaryColon:
                    self.allow_wrap_or_preserved_newline(current_token)
                self._output.space_before_token = True
                return
            elif self._options.operator_position == OPERATOR_POSITION['after_newline']:
                self._output.space_before_token = True
                if not isColon or isTernaryColon:
                    if self._tokens.peek().newlines:
                        self.print_newline(preserve_statement_flags=True)
                    else:
                        self.allow_wrap_or_preserved_newline(current_token)
                else:
                    self._output.space_before_token = False
                self.print_token(current_token)
                self._output.space_before_token = True
                return
            elif self._options.operator_position == OPERATOR_POSITION['preserve_newline']:
                if not isOtherColon:
                    self.allow_wrap_or_preserved_newline(current_token)
                self._output.space_before_token = not (self._output.just_added_newline() or isOtherColon)
                self.print_token(current_token)
                self._output.space_before_token = True
                return
        if isGeneratorAsterisk:
            self.allow_wrap_or_preserved_newline(current_token)
            space_before = False
            next_token = self._tokens.peek()
            space_after = next_token and next_token.type in [TOKEN.WORD, TOKEN.RESERVED]
        elif current_token.text == '...':
            self.allow_wrap_or_preserved_newline(current_token)
            space_before = self._flags.last_token.type == TOKEN.START_BLOCK
            space_after = False
        elif current_token.text in ['--', '++', '!', '~'] or isUnary:
            if self._flags.last_token.type == TOKEN.COMMA or self._flags.last_token.type == TOKEN.START_EXPR:
                self.allow_wrap_or_preserved_newline(current_token)
            space_before = False
            space_after = False
            if current_token.newlines and (current_token.text == '--' or current_token.text == '++' or current_token.text == '~'):
                new_line_needed = reserved_array(self._flags.last_token, _special_word_set) and current_token.newlines
                if new_line_needed and (self._previous_flags.if_block or self._previous_flags.else_block):
                    self.restore_mode()
                self.print_newline(new_line_needed, True)
            if self._flags.last_token.text == ';' and self.is_expression(self._flags.mode):
                space_before = True
            if self._flags.last_token.type == TOKEN.RESERVED:
                space_before = True
            elif self._flags.last_token.type == TOKEN.END_EXPR:
                space_before = not (self._flags.last_token.text == ']' and current_token.text in ['--', '++'])
            elif self._flags.last_token.type == TOKEN.OPERATOR:
                space_before = current_token.text in ['--', '-', '++', '+'] and self._flags.last_token.text in ['--', '-', '++', '+']
                if current_token.text in ['-', '+'] and self._flags.last_token.text in ['--', '++']:
                    space_after = True
            if (self._flags.mode == MODE.BlockStatement and (not self._flags.inline_frame) or self._flags.mode == MODE.Statement) and self._flags.last_token.text in ['{', ';']:
                self.print_newline()
        if space_before:
            self._output.space_before_token = True
        self.print_token(current_token)
        if space_after:
            self._output.space_before_token = True

    def handle_block_comment(self, current_token, preserve_statement_flags):
        if self._output.raw:
            self._output.add_raw_token(current_token)
            if current_token.directives and current_token.directives.get('preserve') == 'end':
                self._output.raw = self._options.test_output_raw
            return
        if current_token.directives:
            self.print_newline(preserve_statement_flags=preserve_statement_flags)
            self.print_token(current_token)
            if current_token.directives.get('preserve') == 'start':
                self._output.raw = True
            self.print_newline(preserve_statement_flags=True)
            return
        if not self.acorn.newline.search(current_token.text) and (not current_token.newlines):
            self._output.space_before_token = True
            self.print_token(current_token)
            self._output.space_before_token = True
            return
        lines = self.acorn.allLineBreaks.split(current_token.text)
        javadoc = False
        starless = False
        last_indent = current_token.whitespace_before
        last_indent_length = len(last_indent)
        self.print_newline(preserve_statement_flags=preserve_statement_flags)
        self.print_token(current_token, lines[0])
        self.print_newline(preserve_statement_flags=preserve_statement_flags)
        if len(lines) > 1:
            lines = lines[1:]
            javadoc = not any((l for l in lines if l.strip() == '' or l.lstrip()[0] != '*'))
            starless = all((l.startswith(last_indent) or l.strip() == '' for l in lines))
            if javadoc:
                self._flags.alignment = 1
            for line in lines:
                if javadoc:
                    self.print_token(current_token, line.lstrip())
                elif starless and len(line) > last_indent_length:
                    self.print_token(current_token, line[last_indent_length:])
                else:
                    self._output.current_line.set_indent(-1)
                    self._output.add_token(line)
                self.print_newline(preserve_statement_flags=preserve_statement_flags)
            self._flags.alignment = 0

    def handle_comment(self, current_token, preserve_statement_flags):
        if current_token.newlines:
            self.print_newline(preserve_statement_flags=preserve_statement_flags)
        if not current_token.newlines:
            self._output.trim(True)
        self._output.space_before_token = True
        self.print_token(current_token)
        self.print_newline(preserve_statement_flags=preserve_statement_flags)

    def handle_dot(self, current_token):
        if self.start_of_statement(current_token):
            pass
        else:
            self.handle_whitespace_and_comments(current_token, True)
        if re.search('^([0-9])+$', self._flags.last_token.text):
            self._output.space_before_token = True
        if reserved_array(self._flags.last_token, _special_word_set):
            self._output.space_before_token = False
        else:
            self.allow_wrap_or_preserved_newline(current_token, self._flags.last_token.text == ')' and self._options.break_chained_methods)
        if self._options.unindent_chained_methods and self._output.just_added_newline():
            self.deindent()
        self.print_token(current_token)

    def handle_unknown(self, current_token, preserve_statement_flags):
        self.print_token(current_token)
        if current_token.text[-1] == '\n':
            self.print_newline(preserve_statement_flags=preserve_statement_flags)

    def handle_eof(self, current_token):
        while self._flags.mode == MODE.Statement:
            self.restore_mode()
        self.handle_whitespace_and_comments(current_token)