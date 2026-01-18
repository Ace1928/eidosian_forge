from __future__ import annotations
import re
import typing
from itertools import product
from typing import Any, Callable
import sympy
from sympy import Mul, Add, Pow, log, exp, sqrt, cos, sin, tan, asin, acos, acot, asec, acsc, sinh, cosh, tanh, asinh, \
from sympy.core.sympify import sympify, _sympify
from sympy.functions.special.bessel import airybiprime
from sympy.functions.special.error_functions import li
from sympy.utilities.exceptions import sympy_deprecation_warning
@_deco
class MathematicaParser:
    """
    An instance of this class converts a string of a Wolfram Mathematica
    expression to a SymPy expression.

    The main parser acts internally in three stages:

    1. tokenizer: tokenizes the Mathematica expression and adds the missing *
        operators. Handled by ``_from_mathematica_to_tokens(...)``
    2. full form list: sort the list of strings output by the tokenizer into a
        syntax tree of nested lists and strings, equivalent to Mathematica's
        ``FullForm`` expression output. This is handled by the function
        ``_from_tokens_to_fullformlist(...)``.
    3. SymPy expression: the syntax tree expressed as full form list is visited
        and the nodes with equivalent classes in SymPy are replaced. Unknown
        syntax tree nodes are cast to SymPy ``Function`` objects. This is
        handled by ``_from_fullformlist_to_sympy(...)``.

    """
    CORRESPONDENCES = {'Sqrt[x]': 'sqrt(x)', 'Exp[x]': 'exp(x)', 'Log[x]': 'log(x)', 'Log[x,y]': 'log(y,x)', 'Log2[x]': 'log(x,2)', 'Log10[x]': 'log(x,10)', 'Mod[x,y]': 'Mod(x,y)', 'Max[*x]': 'Max(*x)', 'Min[*x]': 'Min(*x)', 'Pochhammer[x,y]': 'rf(x,y)', 'ArcTan[x,y]': 'atan2(y,x)', 'ExpIntegralEi[x]': 'Ei(x)', 'SinIntegral[x]': 'Si(x)', 'CosIntegral[x]': 'Ci(x)', 'AiryAi[x]': 'airyai(x)', 'AiryAiPrime[x]': 'airyaiprime(x)', 'AiryBi[x]': 'airybi(x)', 'AiryBiPrime[x]': 'airybiprime(x)', 'LogIntegral[x]': ' li(x)', 'PrimePi[x]': 'primepi(x)', 'Prime[x]': 'prime(x)', 'PrimeQ[x]': 'isprime(x)'}
    for arc, tri, h in product(('', 'Arc'), ('Sin', 'Cos', 'Tan', 'Cot', 'Sec', 'Csc'), ('', 'h')):
        fm = arc + tri + h + '[x]'
        if arc:
            fs = 'a' + tri.lower() + h + '(x)'
        else:
            fs = tri.lower() + h + '(x)'
        CORRESPONDENCES.update({fm: fs})
    REPLACEMENTS = {' ': '', '^': '**', '{': '[', '}': ']'}
    RULES = {'whitespace': (re.compile('\n                (?:(?<=[a-zA-Z\\d])|(?<=\\d\\.))     # a letter or a number\n                \\s+                               # any number of whitespaces\n                (?:(?=[a-zA-Z\\d])|(?=\\.\\d))       # a letter or a number\n                ', re.VERBOSE), '*'), 'add*_1': (re.compile("\n                (?:(?<=[])\\d])|(?<=\\d\\.))       # ], ) or a number\n                                                # ''\n                (?=[(a-zA-Z])                   # ( or a single letter\n                ", re.VERBOSE), '*'), 'add*_2': (re.compile('\n                (?<=[a-zA-Z])       # a letter\n                \\(                  # ( as a character\n                (?=.)               # any characters\n                ', re.VERBOSE), '*('), 'Pi': (re.compile("\n                (?:\n                \\A|(?<=[^a-zA-Z])\n                )\n                Pi                  # 'Pi' is 3.14159... in Mathematica\n                (?=[^a-zA-Z])\n                ", re.VERBOSE), 'pi')}
    FM_PATTERN = re.compile('\n                (?:\n                \\A|(?<=[^a-zA-Z])   # at the top or a non-letter\n                )\n                [A-Z][a-zA-Z\\d]*    # Function\n                (?=\\[)              # [ as a character\n                ', re.VERBOSE)
    ARG_MTRX_PATTERN = re.compile('\n                \\{.*\\}\n                ', re.VERBOSE)
    ARGS_PATTERN_TEMPLATE = '\n                (?:\n                \\A|(?<=[^a-zA-Z])\n                )\n                {arguments}         # model argument like x, y,...\n                (?=[^a-zA-Z])\n                '
    TRANSLATIONS: dict[tuple[str, int], dict[str, Any]] = {}
    cache_original: dict[tuple[str, int], dict[str, Any]] = {}
    cache_compiled: dict[tuple[str, int], dict[str, Any]] = {}

    @classmethod
    def _initialize_class(cls):
        d = cls._compile_dictionary(cls.CORRESPONDENCES)
        cls.TRANSLATIONS.update(d)

    def __init__(self, additional_translations=None):
        self.translations = {}
        self.translations.update(self.TRANSLATIONS)
        if additional_translations is None:
            additional_translations = {}
        if self.__class__.cache_original != additional_translations:
            if not isinstance(additional_translations, dict):
                raise ValueError('The argument must be dict type')
            d = self._compile_dictionary(additional_translations)
            self.__class__.cache_original = additional_translations
            self.__class__.cache_compiled = d
        self.translations.update(self.__class__.cache_compiled)

    @classmethod
    def _compile_dictionary(cls, dic):
        d = {}
        for fm, fs in dic.items():
            cls._check_input(fm)
            cls._check_input(fs)
            fm = cls._apply_rules(fm, 'whitespace')
            fs = cls._apply_rules(fs, 'whitespace')
            fm = cls._replace(fm, ' ')
            fs = cls._replace(fs, ' ')
            m = cls.FM_PATTERN.search(fm)
            if m is None:
                err = "'{f}' function form is invalid.".format(f=fm)
                raise ValueError(err)
            fm_name = m.group()
            args, end = cls._get_args(m)
            if m.start() != 0 or end != len(fm):
                err = "'{f}' function form is invalid.".format(f=fm)
                raise ValueError(err)
            if args[-1][0] == '*':
                key_arg = '*'
            else:
                key_arg = len(args)
            key = (fm_name, key_arg)
            re_args = [x if x[0] != '*' else '\\' + x for x in args]
            xyz = '(?:(' + '|'.join(re_args) + '))'
            patStr = cls.ARGS_PATTERN_TEMPLATE.format(arguments=xyz)
            pat = re.compile(patStr, re.VERBOSE)
            d[key] = {}
            d[key]['fs'] = fs
            d[key]['args'] = args
            d[key]['pat'] = pat
        return d

    def _convert_function(self, s):
        """Parse Mathematica function to SymPy one"""
        pat = self.FM_PATTERN
        scanned = ''
        cur = 0
        while True:
            m = pat.search(s)
            if m is None:
                scanned += s
                break
            fm = m.group()
            args, end = self._get_args(m)
            bgn = m.start()
            s = self._convert_one_function(s, fm, args, bgn, end)
            cur = bgn
            scanned += s[:cur]
            s = s[cur:]
        return scanned

    def _convert_one_function(self, s, fm, args, bgn, end):
        if (fm, len(args)) in self.translations:
            key = (fm, len(args))
            x_args = self.translations[key]['args']
            d = {k: v for k, v in zip(x_args, args)}
        elif (fm, '*') in self.translations:
            key = (fm, '*')
            x_args = self.translations[key]['args']
            d = {}
            for i, x in enumerate(x_args):
                if x[0] == '*':
                    d[x] = ','.join(args[i:])
                    break
                d[x] = args[i]
        else:
            err = "'{f}' is out of the whitelist.".format(f=fm)
            raise ValueError(err)
        template = self.translations[key]['fs']
        pat = self.translations[key]['pat']
        scanned = ''
        cur = 0
        while True:
            m = pat.search(template)
            if m is None:
                scanned += template
                break
            x = m.group()
            xbgn = m.start()
            scanned += template[:xbgn] + d[x]
            cur = m.end()
            template = template[cur:]
        s = s[:bgn] + scanned + s[end:]
        return s

    @classmethod
    def _get_args(cls, m):
        """Get arguments of a Mathematica function"""
        s = m.string
        anc = m.end() + 1
        square, curly = ([], [])
        args = []
        cur = anc
        for i, c in enumerate(s[anc:], anc):
            if c == ',' and (not square) and (not curly):
                args.append(s[cur:i])
                cur = i + 1
            if c == '{':
                curly.append(c)
            elif c == '}':
                curly.pop()
            if c == '[':
                square.append(c)
            elif c == ']':
                if square:
                    square.pop()
                else:
                    args.append(s[cur:i])
                    break
        func_end = i + 1
        return (args, func_end)

    @classmethod
    def _replace(cls, s, bef):
        aft = cls.REPLACEMENTS[bef]
        s = s.replace(bef, aft)
        return s

    @classmethod
    def _apply_rules(cls, s, bef):
        pat, aft = cls.RULES[bef]
        return pat.sub(aft, s)

    @classmethod
    def _check_input(cls, s):
        for bracket in (('[', ']'), ('{', '}'), ('(', ')')):
            if s.count(bracket[0]) != s.count(bracket[1]):
                err = "'{f}' function form is invalid.".format(f=s)
                raise ValueError(err)
        if '{' in s:
            err = 'Currently list is not supported.'
            raise ValueError(err)

    def _parse_old(self, s):
        self._check_input(s)
        s = self._apply_rules(s, 'whitespace')
        s = self._replace(s, ' ')
        s = self._apply_rules(s, 'add*_1')
        s = self._apply_rules(s, 'add*_2')
        s = self._convert_function(s)
        s = self._replace(s, '^')
        s = self._apply_rules(s, 'Pi')
        return s

    def parse(self, s):
        s2 = self._from_mathematica_to_tokens(s)
        s3 = self._from_tokens_to_fullformlist(s2)
        s4 = self._from_fullformlist_to_sympy(s3)
        return s4
    INFIX = 'Infix'
    PREFIX = 'Prefix'
    POSTFIX = 'Postfix'
    FLAT = 'Flat'
    RIGHT = 'Right'
    LEFT = 'Left'
    _mathematica_op_precedence: list[tuple[str, str | None, dict[str, str | Callable]]] = [(POSTFIX, None, {';': lambda x: x + ['Null'] if isinstance(x, list) and x and (x[0] == 'CompoundExpression') else ['CompoundExpression', x, 'Null']}), (INFIX, FLAT, {';': 'CompoundExpression'}), (INFIX, RIGHT, {'=': 'Set', ':=': 'SetDelayed', '+=': 'AddTo', '-=': 'SubtractFrom', '*=': 'TimesBy', '/=': 'DivideBy'}), (INFIX, LEFT, {'//': lambda x, y: [x, y]}), (POSTFIX, None, {'&': 'Function'}), (INFIX, LEFT, {'/.': 'ReplaceAll'}), (INFIX, RIGHT, {'->': 'Rule', ':>': 'RuleDelayed'}), (INFIX, LEFT, {'/;': 'Condition'}), (INFIX, FLAT, {'|': 'Alternatives'}), (POSTFIX, None, {'..': 'Repeated', '...': 'RepeatedNull'}), (INFIX, FLAT, {'||': 'Or'}), (INFIX, FLAT, {'&&': 'And'}), (PREFIX, None, {'!': 'Not'}), (INFIX, FLAT, {'===': 'SameQ', '=!=': 'UnsameQ'}), (INFIX, FLAT, {'==': 'Equal', '!=': 'Unequal', '<=': 'LessEqual', '<': 'Less', '>=': 'GreaterEqual', '>': 'Greater'}), (INFIX, None, {';;': 'Span'}), (INFIX, FLAT, {'+': 'Plus', '-': 'Plus'}), (INFIX, FLAT, {'*': 'Times', '/': 'Times'}), (INFIX, FLAT, {'.': 'Dot'}), (PREFIX, None, {'-': lambda x: MathematicaParser._get_neg(x), '+': lambda x: x}), (INFIX, RIGHT, {'^': 'Power'}), (INFIX, RIGHT, {'@@': 'Apply', '/@': 'Map', '//@': 'MapAll', '@@@': lambda x, y: ['Apply', x, y, ['List', '1']]}), (POSTFIX, None, {"'": 'Derivative', '!': 'Factorial', '!!': 'Factorial2', '--': 'Decrement'}), (INFIX, None, {'[': lambda x, y: [x, *y], '[[': lambda x, y: ['Part', x, *y]}), (PREFIX, None, {'{': lambda x: ['List', *x], '(': lambda x: x[0]}), (INFIX, None, {'?': 'PatternTest'}), (POSTFIX, None, {'_': lambda x: ['Pattern', x, ['Blank']], '_.': lambda x: ['Optional', ['Pattern', x, ['Blank']]], '__': lambda x: ['Pattern', x, ['BlankSequence']], '___': lambda x: ['Pattern', x, ['BlankNullSequence']]}), (INFIX, None, {'_': lambda x, y: ['Pattern', x, ['Blank', y]]}), (PREFIX, None, {'#': 'Slot', '##': 'SlotSequence'})]
    _missing_arguments_default = {'#': lambda: ['Slot', '1'], '##': lambda: ['SlotSequence', '1']}
    _literal = '[A-Za-z][A-Za-z0-9]*'
    _number = '(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)'
    _enclosure_open = ['(', '[', '[[', '{']
    _enclosure_close = [')', ']', ']]', '}']

    @classmethod
    def _get_neg(cls, x):
        return f'-{x}' if isinstance(x, str) and re.match(MathematicaParser._number, x) else ['Times', '-1', x]

    @classmethod
    def _get_inv(cls, x):
        return ['Power', x, '-1']
    _regex_tokenizer = None

    def _get_tokenizer(self):
        if self._regex_tokenizer is not None:
            return self._regex_tokenizer
        tokens = [self._literal, self._number]
        tokens_escape = self._enclosure_open[:] + self._enclosure_close[:]
        for typ, strat, symdict in self._mathematica_op_precedence:
            for k in symdict:
                tokens_escape.append(k)
        tokens_escape.sort(key=lambda x: -len(x))
        tokens.extend(map(re.escape, tokens_escape))
        tokens.append(',')
        tokens.append('\n')
        tokenizer = re.compile('(' + '|'.join(tokens) + ')')
        self._regex_tokenizer = tokenizer
        return self._regex_tokenizer

    def _from_mathematica_to_tokens(self, code: str):
        tokenizer = self._get_tokenizer()
        code_splits: list[str | list] = []
        while True:
            string_start = code.find('"')
            if string_start == -1:
                if len(code) > 0:
                    code_splits.append(code)
                break
            match_end = re.search('(?<!\\\\)"', code[string_start + 1:])
            if match_end is None:
                raise SyntaxError('mismatch in string "  " expression')
            string_end = string_start + match_end.start() + 1
            if string_start > 0:
                code_splits.append(code[:string_start])
            code_splits.append(['_Str', code[string_start + 1:string_end].replace('\\"', '"')])
            code = code[string_end + 1:]
        for i, code_split in enumerate(code_splits):
            if isinstance(code_split, list):
                continue
            while True:
                pos_comment_start = code_split.find('(*')
                if pos_comment_start == -1:
                    break
                pos_comment_end = code_split.find('*)')
                if pos_comment_end == -1 or pos_comment_end < pos_comment_start:
                    raise SyntaxError('mismatch in comment (*  *) code')
                code_split = code_split[:pos_comment_start] + code_split[pos_comment_end + 2:]
            code_splits[i] = code_split
        token_lists = [tokenizer.findall(i) if isinstance(i, str) and i.isascii() else [i] for i in code_splits]
        tokens = [j for i in token_lists for j in i]
        while tokens and tokens[0] == '\n':
            tokens.pop(0)
        while tokens and tokens[-1] == '\n':
            tokens.pop(-1)
        return tokens

    def _is_op(self, token: str | list) -> bool:
        if isinstance(token, list):
            return False
        if re.match(self._literal, token):
            return False
        if re.match('-?' + self._number, token):
            return False
        return True

    def _is_valid_star1(self, token: str | list) -> bool:
        if token in (')', '}'):
            return True
        return not self._is_op(token)

    def _is_valid_star2(self, token: str | list) -> bool:
        if token in ('(', '{'):
            return True
        return not self._is_op(token)

    def _from_tokens_to_fullformlist(self, tokens: list):
        stack: list[list] = [[]]
        open_seq = []
        pointer: int = 0
        while pointer < len(tokens):
            token = tokens[pointer]
            if token in self._enclosure_open:
                stack[-1].append(token)
                open_seq.append(token)
                stack.append([])
            elif token == ',':
                if len(stack[-1]) == 0 and stack[-2][-1] == open_seq[-1]:
                    raise SyntaxError('%s cannot be followed by comma ,' % open_seq[-1])
                stack[-1] = self._parse_after_braces(stack[-1])
                stack.append([])
            elif token in self._enclosure_close:
                ind = self._enclosure_close.index(token)
                if self._enclosure_open[ind] != open_seq[-1]:
                    unmatched_enclosure = SyntaxError('unmatched enclosure')
                    if token == ']]' and open_seq[-1] == '[':
                        if open_seq[-2] == '[':
                            tokens.insert(pointer + 1, ']')
                        elif open_seq[-2] == '[[':
                            if tokens[pointer + 1] == ']':
                                tokens[pointer + 1] = ']]'
                            elif tokens[pointer + 1] == ']]':
                                tokens[pointer + 1] = ']]'
                                tokens.insert(pointer + 2, ']')
                            else:
                                raise unmatched_enclosure
                    else:
                        raise unmatched_enclosure
                if len(stack[-1]) == 0 and stack[-2][-1] == '(':
                    raise SyntaxError('( ) not valid syntax')
                last_stack = self._parse_after_braces(stack[-1], True)
                stack[-1] = last_stack
                new_stack_element = []
                while stack[-1][-1] != open_seq[-1]:
                    new_stack_element.append(stack.pop())
                new_stack_element.reverse()
                if open_seq[-1] == '(' and len(new_stack_element) != 1:
                    raise SyntaxError('( must be followed by one expression, %i detected' % len(new_stack_element))
                stack[-1].append(new_stack_element)
                open_seq.pop(-1)
            else:
                stack[-1].append(token)
            pointer += 1
        assert len(stack) == 1
        return self._parse_after_braces(stack[0])

    def _util_remove_newlines(self, lines: list, tokens: list, inside_enclosure: bool):
        pointer = 0
        size = len(tokens)
        while pointer < size:
            token = tokens[pointer]
            if token == '\n':
                if inside_enclosure:
                    tokens.pop(pointer)
                    size -= 1
                    continue
                if pointer == 0:
                    tokens.pop(0)
                    size -= 1
                    continue
                if pointer > 1:
                    try:
                        prev_expr = self._parse_after_braces(tokens[:pointer], inside_enclosure)
                    except SyntaxError:
                        tokens.pop(pointer)
                        size -= 1
                        continue
                else:
                    prev_expr = tokens[0]
                if len(prev_expr) > 0 and prev_expr[0] == 'CompoundExpression':
                    lines.extend(prev_expr[1:])
                else:
                    lines.append(prev_expr)
                for i in range(pointer):
                    tokens.pop(0)
                size -= pointer
                pointer = 0
                continue
            pointer += 1

    def _util_add_missing_asterisks(self, tokens: list):
        size: int = len(tokens)
        pointer: int = 0
        while pointer < size:
            if pointer > 0 and self._is_valid_star1(tokens[pointer - 1]) and self._is_valid_star2(tokens[pointer]):
                if tokens[pointer] == '(':
                    tokens[pointer] = '*'
                    tokens[pointer + 1] = tokens[pointer + 1][0]
                else:
                    tokens.insert(pointer, '*')
                    pointer += 1
                    size += 1
            pointer += 1

    def _parse_after_braces(self, tokens: list, inside_enclosure: bool=False):
        op_dict: dict
        changed: bool = False
        lines: list = []
        self._util_remove_newlines(lines, tokens, inside_enclosure)
        for op_type, grouping_strat, op_dict in reversed(self._mathematica_op_precedence):
            if '*' in op_dict:
                self._util_add_missing_asterisks(tokens)
            size: int = len(tokens)
            pointer: int = 0
            while pointer < size:
                token = tokens[pointer]
                if isinstance(token, str) and token in op_dict:
                    op_name: str | Callable = op_dict[token]
                    node: list
                    first_index: int
                    if isinstance(op_name, str):
                        node = [op_name]
                        first_index = 1
                    else:
                        node = []
                        first_index = 0
                    if token in ('+', '-') and op_type == self.PREFIX and (pointer > 0) and (not self._is_op(tokens[pointer - 1])):
                        pointer += 1
                        continue
                    if op_type == self.INFIX:
                        if pointer == 0 or pointer == size - 1 or self._is_op(tokens[pointer - 1]) or self._is_op(tokens[pointer + 1]):
                            pointer += 1
                            continue
                    changed = True
                    tokens[pointer] = node
                    if op_type == self.INFIX:
                        arg1 = tokens.pop(pointer - 1)
                        arg2 = tokens.pop(pointer)
                        if token == '/':
                            arg2 = self._get_inv(arg2)
                        elif token == '-':
                            arg2 = self._get_neg(arg2)
                        pointer -= 1
                        size -= 2
                        node.append(arg1)
                        node_p = node
                        if grouping_strat == self.FLAT:
                            while pointer + 2 < size and self._check_op_compatible(tokens[pointer + 1], token):
                                node_p.append(arg2)
                                other_op = tokens.pop(pointer + 1)
                                arg2 = tokens.pop(pointer + 1)
                                if other_op == '/':
                                    arg2 = self._get_inv(arg2)
                                elif other_op == '-':
                                    arg2 = self._get_neg(arg2)
                                size -= 2
                            node_p.append(arg2)
                        elif grouping_strat == self.RIGHT:
                            while pointer + 2 < size and tokens[pointer + 1] == token:
                                node_p.append([op_name, arg2])
                                node_p = node_p[-1]
                                tokens.pop(pointer + 1)
                                arg2 = tokens.pop(pointer + 1)
                                size -= 2
                            node_p.append(arg2)
                        elif grouping_strat == self.LEFT:
                            while pointer + 1 < size and tokens[pointer + 1] == token:
                                if isinstance(op_name, str):
                                    node_p[first_index] = [op_name, node_p[first_index], arg2]
                                else:
                                    node_p[first_index] = op_name(node_p[first_index], arg2)
                                tokens.pop(pointer + 1)
                                arg2 = tokens.pop(pointer + 1)
                                size -= 2
                            node_p.append(arg2)
                        else:
                            node.append(arg2)
                    elif op_type == self.PREFIX:
                        assert grouping_strat is None
                        if pointer == size - 1 or self._is_op(tokens[pointer + 1]):
                            tokens[pointer] = self._missing_arguments_default[token]()
                        else:
                            node.append(tokens.pop(pointer + 1))
                            size -= 1
                    elif op_type == self.POSTFIX:
                        assert grouping_strat is None
                        if pointer == 0 or self._is_op(tokens[pointer - 1]):
                            tokens[pointer] = self._missing_arguments_default[token]()
                        else:
                            node.append(tokens.pop(pointer - 1))
                            pointer -= 1
                            size -= 1
                    if isinstance(op_name, Callable):
                        op_call: Callable = typing.cast(Callable, op_name)
                        new_node = op_call(*node)
                        node.clear()
                        if isinstance(new_node, list):
                            node.extend(new_node)
                        else:
                            tokens[pointer] = new_node
                pointer += 1
        if len(tokens) > 1 or (len(lines) == 0 and len(tokens) == 0):
            if changed:
                return self._parse_after_braces(tokens, inside_enclosure)
            raise SyntaxError('unable to create a single AST for the expression')
        if len(lines) > 0:
            if tokens[0] and tokens[0][0] == 'CompoundExpression':
                tokens = tokens[0][1:]
            compound_expression = ['CompoundExpression', *lines, *tokens]
            return compound_expression
        return tokens[0]

    def _check_op_compatible(self, op1: str, op2: str):
        if op1 == op2:
            return True
        muldiv = {'*', '/'}
        addsub = {'+', '-'}
        if op1 in muldiv and op2 in muldiv:
            return True
        if op1 in addsub and op2 in addsub:
            return True
        return False

    def _from_fullform_to_fullformlist(self, wmexpr: str):
        """
        Parses FullForm[Downvalues[]] generated by Mathematica
        """
        out: list = []
        stack = [out]
        generator = re.finditer('[\\[\\],]', wmexpr)
        last_pos = 0
        for match in generator:
            if match is None:
                break
            position = match.start()
            last_expr = wmexpr[last_pos:position].replace(',', '').replace(']', '').replace('[', '').strip()
            if match.group() == ',':
                if last_expr != '':
                    stack[-1].append(last_expr)
            elif match.group() == ']':
                if last_expr != '':
                    stack[-1].append(last_expr)
                stack.pop()
            elif match.group() == '[':
                stack[-1].append([last_expr])
                stack.append(stack[-1][-1])
            last_pos = match.end()
        return out[0]

    def _from_fullformlist_to_fullformsympy(self, pylist: list):
        from sympy import Function, Symbol

        def converter(expr):
            if isinstance(expr, list):
                if len(expr) > 0:
                    head = expr[0]
                    args = [converter(arg) for arg in expr[1:]]
                    return Function(head)(*args)
                else:
                    raise ValueError('Empty list of expressions')
            elif isinstance(expr, str):
                return Symbol(expr)
            else:
                return _sympify(expr)
        return converter(pylist)
    _node_conversions = {'Times': Mul, 'Plus': Add, 'Power': Pow, 'Log': lambda *a: log(*reversed(a)), 'Log2': lambda x: log(x, 2), 'Log10': lambda x: log(x, 10), 'Exp': exp, 'Sqrt': sqrt, 'Sin': sin, 'Cos': cos, 'Tan': tan, 'Cot': cot, 'Sec': sec, 'Csc': csc, 'ArcSin': asin, 'ArcCos': acos, 'ArcTan': lambda *a: atan2(*reversed(a)) if len(a) == 2 else atan(*a), 'ArcCot': acot, 'ArcSec': asec, 'ArcCsc': acsc, 'Sinh': sinh, 'Cosh': cosh, 'Tanh': tanh, 'Coth': coth, 'Sech': sech, 'Csch': csch, 'ArcSinh': asinh, 'ArcCosh': acosh, 'ArcTanh': atanh, 'ArcCoth': acoth, 'ArcSech': asech, 'ArcCsch': acsch, 'Expand': expand, 'Im': im, 'Re': sympy.re, 'Flatten': flatten, 'Polylog': polylog, 'Cancel': cancel, 'TrigExpand': expand_trig, 'Sign': sign, 'Simplify': simplify, 'Defer': UnevaluatedExpr, 'Identity': S, 'Null': lambda *a: S.Zero, 'Mod': Mod, 'Max': Max, 'Min': Min, 'Pochhammer': rf, 'ExpIntegralEi': Ei, 'SinIntegral': Si, 'CosIntegral': Ci, 'AiryAi': airyai, 'AiryAiPrime': airyaiprime, 'AiryBi': airybi, 'AiryBiPrime': airybiprime, 'LogIntegral': li, 'PrimePi': primepi, 'Prime': prime, 'PrimeQ': isprime, 'List': Tuple, 'Greater': StrictGreaterThan, 'GreaterEqual': GreaterThan, 'Less': StrictLessThan, 'LessEqual': LessThan, 'Equal': Equality, 'Or': Or, 'And': And, 'Function': _parse_Function}
    _atom_conversions = {'I': I, 'Pi': pi}

    def _from_fullformlist_to_sympy(self, full_form_list):

        def recurse(expr):
            if isinstance(expr, list):
                if isinstance(expr[0], list):
                    head = recurse(expr[0])
                else:
                    head = self._node_conversions.get(expr[0], Function(expr[0]))
                return head(*[recurse(arg) for arg in expr[1:]])
            else:
                return self._atom_conversions.get(expr, sympify(expr))
        return recurse(full_form_list)

    def _from_fullformsympy_to_sympy(self, mform):
        expr = mform
        for mma_form, sympy_node in self._node_conversions.items():
            expr = expr.replace(Function(mma_form), sympy_node)
        return expr