import re
import types
import sys
import os.path
import inspect
import base64
import warnings
class ParserReflect(object):

    def __init__(self, pdict, log=None):
        self.pdict = pdict
        self.start = None
        self.error_func = None
        self.tokens = None
        self.modules = set()
        self.grammar = []
        self.error = False
        if log is None:
            self.log = PlyLogger(sys.stderr)
        else:
            self.log = log

    def get_all(self):
        self.get_start()
        self.get_error_func()
        self.get_tokens()
        self.get_precedence()
        self.get_pfunctions()

    def validate_all(self):
        self.validate_start()
        self.validate_error_func()
        self.validate_tokens()
        self.validate_precedence()
        self.validate_pfunctions()
        self.validate_modules()
        return self.error

    def signature(self):
        parts = []
        try:
            if self.start:
                parts.append(self.start)
            if self.prec:
                parts.append(''.join([''.join(p) for p in self.prec]))
            if self.tokens:
                parts.append(' '.join(self.tokens))
            for f in self.pfuncs:
                if f[3]:
                    parts.append(f[3])
        except (TypeError, ValueError):
            pass
        return ''.join(parts)

    def validate_modules(self):
        fre = re.compile('\\s*def\\s+(p_[a-zA-Z_0-9]*)\\(')
        for module in self.modules:
            try:
                lines, linen = inspect.getsourcelines(module)
            except IOError:
                continue
            counthash = {}
            for linen, line in enumerate(lines):
                linen += 1
                m = fre.match(line)
                if m:
                    name = m.group(1)
                    prev = counthash.get(name)
                    if not prev:
                        counthash[name] = linen
                    else:
                        filename = inspect.getsourcefile(module)
                        self.log.warning('%s:%d: Function %s redefined. Previously defined on line %d', filename, linen, name, prev)

    def get_start(self):
        self.start = self.pdict.get('start')

    def validate_start(self):
        if self.start is not None:
            if not isinstance(self.start, string_types):
                self.log.error("'start' must be a string")

    def get_error_func(self):
        self.error_func = self.pdict.get('p_error')

    def validate_error_func(self):
        if self.error_func:
            if isinstance(self.error_func, types.FunctionType):
                ismethod = 0
            elif isinstance(self.error_func, types.MethodType):
                ismethod = 1
            else:
                self.log.error("'p_error' defined, but is not a function or method")
                self.error = True
                return
            eline = self.error_func.__code__.co_firstlineno
            efile = self.error_func.__code__.co_filename
            module = inspect.getmodule(self.error_func)
            self.modules.add(module)
            argcount = self.error_func.__code__.co_argcount - ismethod
            if argcount != 1:
                self.log.error('%s:%d: p_error() requires 1 argument', efile, eline)
                self.error = True

    def get_tokens(self):
        tokens = self.pdict.get('tokens')
        if not tokens:
            self.log.error('No token list is defined')
            self.error = True
            return
        if not isinstance(tokens, (list, tuple)):
            self.log.error('tokens must be a list or tuple')
            self.error = True
            return
        if not tokens:
            self.log.error('tokens is empty')
            self.error = True
            return
        self.tokens = tokens

    def validate_tokens(self):
        if 'error' in self.tokens:
            self.log.error("Illegal token name 'error'. Is a reserved word")
            self.error = True
            return
        terminals = set()
        for n in self.tokens:
            if n in terminals:
                self.log.warning('Token %r multiply defined', n)
            terminals.add(n)

    def get_precedence(self):
        self.prec = self.pdict.get('precedence')

    def validate_precedence(self):
        preclist = []
        if self.prec:
            if not isinstance(self.prec, (list, tuple)):
                self.log.error('precedence must be a list or tuple')
                self.error = True
                return
            for level, p in enumerate(self.prec):
                if not isinstance(p, (list, tuple)):
                    self.log.error('Bad precedence table')
                    self.error = True
                    return
                if len(p) < 2:
                    self.log.error('Malformed precedence entry %s. Must be (assoc, term, ..., term)', p)
                    self.error = True
                    return
                assoc = p[0]
                if not isinstance(assoc, string_types):
                    self.log.error('precedence associativity must be a string')
                    self.error = True
                    return
                for term in p[1:]:
                    if not isinstance(term, string_types):
                        self.log.error('precedence items must be strings')
                        self.error = True
                        return
                    preclist.append((term, assoc, level + 1))
        self.preclist = preclist

    def get_pfunctions(self):
        p_functions = []
        for name, item in self.pdict.items():
            if not name.startswith('p_') or name == 'p_error':
                continue
            if isinstance(item, (types.FunctionType, types.MethodType)):
                line = getattr(item, 'co_firstlineno', item.__code__.co_firstlineno)
                module = inspect.getmodule(item)
                p_functions.append((line, module, name, item.__doc__))
        p_functions.sort(key=lambda p_function: (p_function[0], str(p_function[1]), p_function[2], p_function[3]))
        self.pfuncs = p_functions

    def validate_pfunctions(self):
        grammar = []
        if len(self.pfuncs) == 0:
            self.log.error('no rules of the form p_rulename are defined')
            self.error = True
            return
        for line, module, name, doc in self.pfuncs:
            file = inspect.getsourcefile(module)
            func = self.pdict[name]
            if isinstance(func, types.MethodType):
                reqargs = 2
            else:
                reqargs = 1
            if func.__code__.co_argcount > reqargs:
                self.log.error('%s:%d: Rule %r has too many arguments', file, line, func.__name__)
                self.error = True
            elif func.__code__.co_argcount < reqargs:
                self.log.error('%s:%d: Rule %r requires an argument', file, line, func.__name__)
                self.error = True
            elif not func.__doc__:
                self.log.warning('%s:%d: No documentation string specified in function %r (ignored)', file, line, func.__name__)
            else:
                try:
                    parsed_g = parse_grammar(doc, file, line)
                    for g in parsed_g:
                        grammar.append((name, g))
                except SyntaxError as e:
                    self.log.error(str(e))
                    self.error = True
                self.modules.add(module)
        for n, v in self.pdict.items():
            if n.startswith('p_') and isinstance(v, (types.FunctionType, types.MethodType)):
                continue
            if n.startswith('t_'):
                continue
            if n.startswith('p_') and n != 'p_error':
                self.log.warning('%r not defined as a function', n)
            if isinstance(v, types.FunctionType) and v.__code__.co_argcount == 1 or (isinstance(v, types.MethodType) and v.__func__.__code__.co_argcount == 2):
                if v.__doc__:
                    try:
                        doc = v.__doc__.split(' ')
                        if doc[1] == ':':
                            self.log.warning('%s:%d: Possible grammar rule %r defined without p_ prefix', v.__code__.co_filename, v.__code__.co_firstlineno, n)
                    except IndexError:
                        pass
        self.grammar = grammar