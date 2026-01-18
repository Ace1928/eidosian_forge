import re
import types
import sys
import os.path
import inspect
import base64
import warnings
class LRParser:

    def __init__(self, lrtab, errorf):
        self.productions = lrtab.lr_productions
        self.action = lrtab.lr_action
        self.goto = lrtab.lr_goto
        self.errorfunc = errorf
        self.set_defaulted_states()
        self.errorok = True

    def errok(self):
        self.errorok = True

    def restart(self):
        del self.statestack[:]
        del self.symstack[:]
        sym = YaccSymbol()
        sym.type = '$end'
        self.symstack.append(sym)
        self.statestack.append(0)

    def set_defaulted_states(self):
        self.defaulted_states = {}
        for state, actions in self.action.items():
            rules = list(actions.values())
            if len(rules) == 1 and rules[0] < 0:
                self.defaulted_states[state] = rules[0]

    def disable_defaulted_states(self):
        self.defaulted_states = {}

    def parse(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
        if debug or yaccdevel:
            if isinstance(debug, int):
                debug = PlyLogger(sys.stderr)
            return self.parsedebug(input, lexer, debug, tracking, tokenfunc)
        elif tracking:
            return self.parseopt(input, lexer, debug, tracking, tokenfunc)
        else:
            return self.parseopt_notrack(input, lexer, debug, tracking, tokenfunc)

    def parsedebug(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
        lookahead = None
        lookaheadstack = []
        actions = self.action
        goto = self.goto
        prod = self.productions
        defaulted_states = self.defaulted_states
        pslice = YaccProduction(None)
        errorcount = 0
        debug.info('PLY: PARSE DEBUG START')
        if not lexer:
            from . import lex
            lexer = lex.lexer
        pslice.lexer = lexer
        pslice.parser = self
        if input is not None:
            lexer.input(input)
        if tokenfunc is None:
            get_token = lexer.token
        else:
            get_token = tokenfunc
        self.token = get_token
        statestack = []
        self.statestack = statestack
        symstack = []
        self.symstack = symstack
        pslice.stack = symstack
        errtoken = None
        statestack.append(0)
        sym = YaccSymbol()
        sym.type = '$end'
        symstack.append(sym)
        state = 0
        while True:
            debug.debug('')
            debug.debug('State  : %s', state)
            if state not in defaulted_states:
                if not lookahead:
                    if not lookaheadstack:
                        lookahead = get_token()
                    else:
                        lookahead = lookaheadstack.pop()
                    if not lookahead:
                        lookahead = YaccSymbol()
                        lookahead.type = '$end'
                ltype = lookahead.type
                t = actions[state].get(ltype)
            else:
                t = defaulted_states[state]
                debug.debug('Defaulted state %s: Reduce using %d', state, -t)
            debug.debug('Stack  : %s', ('%s . %s' % (' '.join([xx.type for xx in symstack][1:]), str(lookahead))).lstrip())
            if t is not None:
                if t > 0:
                    statestack.append(t)
                    state = t
                    debug.debug('Action : Shift and goto state %s', t)
                    symstack.append(lookahead)
                    lookahead = None
                    if errorcount:
                        errorcount -= 1
                    continue
                if t < 0:
                    p = prod[-t]
                    pname = p.name
                    plen = p.len
                    sym = YaccSymbol()
                    sym.type = pname
                    sym.value = None
                    if plen:
                        debug.info('Action : Reduce rule [%s] with %s and goto state %d', p.str, '[' + ','.join([format_stack_entry(_v.value) for _v in symstack[-plen:]]) + ']', goto[statestack[-1 - plen]][pname])
                    else:
                        debug.info('Action : Reduce rule [%s] with %s and goto state %d', p.str, [], goto[statestack[-1]][pname])
                    if plen:
                        targ = symstack[-plen - 1:]
                        targ[0] = sym
                        if tracking:
                            t1 = targ[1]
                            sym.lineno = t1.lineno
                            sym.lexpos = t1.lexpos
                            t1 = targ[-1]
                            sym.endlineno = getattr(t1, 'endlineno', t1.lineno)
                            sym.endlexpos = getattr(t1, 'endlexpos', t1.lexpos)
                        pslice.slice = targ
                        try:
                            del symstack[-plen:]
                            self.state = state
                            p.callable(pslice)
                            del statestack[-plen:]
                            debug.info('Result : %s', format_result(pslice[0]))
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            lookaheadstack.append(lookahead)
                            symstack.extend(targ[1:-1])
                            statestack.pop()
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False
                        continue
                    else:
                        if tracking:
                            sym.lineno = lexer.lineno
                            sym.lexpos = lexer.lexpos
                        targ = [sym]
                        pslice.slice = targ
                        try:
                            self.state = state
                            p.callable(pslice)
                            debug.info('Result : %s', format_result(pslice[0]))
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            lookaheadstack.append(lookahead)
                            statestack.pop()
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False
                        continue
                if t == 0:
                    n = symstack[-1]
                    result = getattr(n, 'value', None)
                    debug.info('Done   : Returning %s', format_result(result))
                    debug.info('PLY: PARSE DEBUG END')
                    return result
            if t is None:
                debug.error('Error  : %s', ('%s . %s' % (' '.join([xx.type for xx in symstack][1:]), str(lookahead))).lstrip())
                if errorcount == 0 or self.errorok:
                    errorcount = error_count
                    self.errorok = False
                    errtoken = lookahead
                    if errtoken.type == '$end':
                        errtoken = None
                    if self.errorfunc:
                        if errtoken and (not hasattr(errtoken, 'lexer')):
                            errtoken.lexer = lexer
                        self.state = state
                        tok = call_errorfunc(self.errorfunc, errtoken, self)
                        if self.errorok:
                            lookahead = tok
                            errtoken = None
                            continue
                    elif errtoken:
                        if hasattr(errtoken, 'lineno'):
                            lineno = lookahead.lineno
                        else:
                            lineno = 0
                        if lineno:
                            sys.stderr.write('yacc: Syntax error at line %d, token=%s\n' % (lineno, errtoken.type))
                        else:
                            sys.stderr.write('yacc: Syntax error, token=%s' % errtoken.type)
                    else:
                        sys.stderr.write('yacc: Parse error in input. EOF\n')
                        return
                else:
                    errorcount = error_count
                if len(statestack) <= 1 and lookahead.type != '$end':
                    lookahead = None
                    errtoken = None
                    state = 0
                    del lookaheadstack[:]
                    continue
                if lookahead.type == '$end':
                    return
                if lookahead.type != 'error':
                    sym = symstack[-1]
                    if sym.type == 'error':
                        if tracking:
                            sym.endlineno = getattr(lookahead, 'lineno', sym.lineno)
                            sym.endlexpos = getattr(lookahead, 'lexpos', sym.lexpos)
                        lookahead = None
                        continue
                    t = YaccSymbol()
                    t.type = 'error'
                    if hasattr(lookahead, 'lineno'):
                        t.lineno = t.endlineno = lookahead.lineno
                    if hasattr(lookahead, 'lexpos'):
                        t.lexpos = t.endlexpos = lookahead.lexpos
                    t.value = lookahead
                    lookaheadstack.append(lookahead)
                    lookahead = t
                else:
                    sym = symstack.pop()
                    if tracking:
                        lookahead.lineno = sym.lineno
                        lookahead.lexpos = sym.lexpos
                    statestack.pop()
                    state = statestack[-1]
                continue
            raise RuntimeError('yacc: internal parser error!!!\n')

    def parseopt(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
        lookahead = None
        lookaheadstack = []
        actions = self.action
        goto = self.goto
        prod = self.productions
        defaulted_states = self.defaulted_states
        pslice = YaccProduction(None)
        errorcount = 0
        if not lexer:
            from . import lex
            lexer = lex.lexer
        pslice.lexer = lexer
        pslice.parser = self
        if input is not None:
            lexer.input(input)
        if tokenfunc is None:
            get_token = lexer.token
        else:
            get_token = tokenfunc
        self.token = get_token
        statestack = []
        self.statestack = statestack
        symstack = []
        self.symstack = symstack
        pslice.stack = symstack
        errtoken = None
        statestack.append(0)
        sym = YaccSymbol()
        sym.type = '$end'
        symstack.append(sym)
        state = 0
        while True:
            if state not in defaulted_states:
                if not lookahead:
                    if not lookaheadstack:
                        lookahead = get_token()
                    else:
                        lookahead = lookaheadstack.pop()
                    if not lookahead:
                        lookahead = YaccSymbol()
                        lookahead.type = '$end'
                ltype = lookahead.type
                t = actions[state].get(ltype)
            else:
                t = defaulted_states[state]
            if t is not None:
                if t > 0:
                    statestack.append(t)
                    state = t
                    symstack.append(lookahead)
                    lookahead = None
                    if errorcount:
                        errorcount -= 1
                    continue
                if t < 0:
                    p = prod[-t]
                    pname = p.name
                    plen = p.len
                    sym = YaccSymbol()
                    sym.type = pname
                    sym.value = None
                    if plen:
                        targ = symstack[-plen - 1:]
                        targ[0] = sym
                        if tracking:
                            t1 = targ[1]
                            sym.lineno = t1.lineno
                            sym.lexpos = t1.lexpos
                            t1 = targ[-1]
                            sym.endlineno = getattr(t1, 'endlineno', t1.lineno)
                            sym.endlexpos = getattr(t1, 'endlexpos', t1.lexpos)
                        pslice.slice = targ
                        try:
                            del symstack[-plen:]
                            self.state = state
                            p.callable(pslice)
                            del statestack[-plen:]
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            lookaheadstack.append(lookahead)
                            symstack.extend(targ[1:-1])
                            statestack.pop()
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False
                        continue
                    else:
                        if tracking:
                            sym.lineno = lexer.lineno
                            sym.lexpos = lexer.lexpos
                        targ = [sym]
                        pslice.slice = targ
                        try:
                            self.state = state
                            p.callable(pslice)
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            lookaheadstack.append(lookahead)
                            statestack.pop()
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False
                        continue
                if t == 0:
                    n = symstack[-1]
                    result = getattr(n, 'value', None)
                    return result
            if t is None:
                if errorcount == 0 or self.errorok:
                    errorcount = error_count
                    self.errorok = False
                    errtoken = lookahead
                    if errtoken.type == '$end':
                        errtoken = None
                    if self.errorfunc:
                        if errtoken and (not hasattr(errtoken, 'lexer')):
                            errtoken.lexer = lexer
                        self.state = state
                        tok = call_errorfunc(self.errorfunc, errtoken, self)
                        if self.errorok:
                            lookahead = tok
                            errtoken = None
                            continue
                    elif errtoken:
                        if hasattr(errtoken, 'lineno'):
                            lineno = lookahead.lineno
                        else:
                            lineno = 0
                        if lineno:
                            sys.stderr.write('yacc: Syntax error at line %d, token=%s\n' % (lineno, errtoken.type))
                        else:
                            sys.stderr.write('yacc: Syntax error, token=%s' % errtoken.type)
                    else:
                        sys.stderr.write('yacc: Parse error in input. EOF\n')
                        return
                else:
                    errorcount = error_count
                if len(statestack) <= 1 and lookahead.type != '$end':
                    lookahead = None
                    errtoken = None
                    state = 0
                    del lookaheadstack[:]
                    continue
                if lookahead.type == '$end':
                    return
                if lookahead.type != 'error':
                    sym = symstack[-1]
                    if sym.type == 'error':
                        if tracking:
                            sym.endlineno = getattr(lookahead, 'lineno', sym.lineno)
                            sym.endlexpos = getattr(lookahead, 'lexpos', sym.lexpos)
                        lookahead = None
                        continue
                    t = YaccSymbol()
                    t.type = 'error'
                    if hasattr(lookahead, 'lineno'):
                        t.lineno = t.endlineno = lookahead.lineno
                    if hasattr(lookahead, 'lexpos'):
                        t.lexpos = t.endlexpos = lookahead.lexpos
                    t.value = lookahead
                    lookaheadstack.append(lookahead)
                    lookahead = t
                else:
                    sym = symstack.pop()
                    if tracking:
                        lookahead.lineno = sym.lineno
                        lookahead.lexpos = sym.lexpos
                    statestack.pop()
                    state = statestack[-1]
                continue
            raise RuntimeError('yacc: internal parser error!!!\n')

    def parseopt_notrack(self, input=None, lexer=None, debug=False, tracking=False, tokenfunc=None):
        lookahead = None
        lookaheadstack = []
        actions = self.action
        goto = self.goto
        prod = self.productions
        defaulted_states = self.defaulted_states
        pslice = YaccProduction(None)
        errorcount = 0
        if not lexer:
            from . import lex
            lexer = lex.lexer
        pslice.lexer = lexer
        pslice.parser = self
        if input is not None:
            lexer.input(input)
        if tokenfunc is None:
            get_token = lexer.token
        else:
            get_token = tokenfunc
        self.token = get_token
        statestack = []
        self.statestack = statestack
        symstack = []
        self.symstack = symstack
        pslice.stack = symstack
        errtoken = None
        statestack.append(0)
        sym = YaccSymbol()
        sym.type = '$end'
        symstack.append(sym)
        state = 0
        while True:
            if state not in defaulted_states:
                if not lookahead:
                    if not lookaheadstack:
                        lookahead = get_token()
                    else:
                        lookahead = lookaheadstack.pop()
                    if not lookahead:
                        lookahead = YaccSymbol()
                        lookahead.type = '$end'
                ltype = lookahead.type
                t = actions[state].get(ltype)
            else:
                t = defaulted_states[state]
            if t is not None:
                if t > 0:
                    statestack.append(t)
                    state = t
                    symstack.append(lookahead)
                    lookahead = None
                    if errorcount:
                        errorcount -= 1
                    continue
                if t < 0:
                    p = prod[-t]
                    pname = p.name
                    plen = p.len
                    sym = YaccSymbol()
                    sym.type = pname
                    sym.value = None
                    if plen:
                        targ = symstack[-plen - 1:]
                        targ[0] = sym
                        pslice.slice = targ
                        try:
                            del symstack[-plen:]
                            self.state = state
                            p.callable(pslice)
                            del statestack[-plen:]
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            lookaheadstack.append(lookahead)
                            symstack.extend(targ[1:-1])
                            statestack.pop()
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False
                        continue
                    else:
                        targ = [sym]
                        pslice.slice = targ
                        try:
                            self.state = state
                            p.callable(pslice)
                            symstack.append(sym)
                            state = goto[statestack[-1]][pname]
                            statestack.append(state)
                        except SyntaxError:
                            lookaheadstack.append(lookahead)
                            statestack.pop()
                            state = statestack[-1]
                            sym.type = 'error'
                            sym.value = 'error'
                            lookahead = sym
                            errorcount = error_count
                            self.errorok = False
                        continue
                if t == 0:
                    n = symstack[-1]
                    result = getattr(n, 'value', None)
                    return result
            if t is None:
                if errorcount == 0 or self.errorok:
                    errorcount = error_count
                    self.errorok = False
                    errtoken = lookahead
                    if errtoken.type == '$end':
                        errtoken = None
                    if self.errorfunc:
                        if errtoken and (not hasattr(errtoken, 'lexer')):
                            errtoken.lexer = lexer
                        self.state = state
                        tok = call_errorfunc(self.errorfunc, errtoken, self)
                        if self.errorok:
                            lookahead = tok
                            errtoken = None
                            continue
                    elif errtoken:
                        if hasattr(errtoken, 'lineno'):
                            lineno = lookahead.lineno
                        else:
                            lineno = 0
                        if lineno:
                            sys.stderr.write('yacc: Syntax error at line %d, token=%s\n' % (lineno, errtoken.type))
                        else:
                            sys.stderr.write('yacc: Syntax error, token=%s' % errtoken.type)
                    else:
                        sys.stderr.write('yacc: Parse error in input. EOF\n')
                        return
                else:
                    errorcount = error_count
                if len(statestack) <= 1 and lookahead.type != '$end':
                    lookahead = None
                    errtoken = None
                    state = 0
                    del lookaheadstack[:]
                    continue
                if lookahead.type == '$end':
                    return
                if lookahead.type != 'error':
                    sym = symstack[-1]
                    if sym.type == 'error':
                        lookahead = None
                        continue
                    t = YaccSymbol()
                    t.type = 'error'
                    if hasattr(lookahead, 'lineno'):
                        t.lineno = t.endlineno = lookahead.lineno
                    if hasattr(lookahead, 'lexpos'):
                        t.lexpos = t.endlexpos = lookahead.lexpos
                    t.value = lookahead
                    lookaheadstack.append(lookahead)
                    lookahead = t
                else:
                    sym = symstack.pop()
                    statestack.pop()
                    state = statestack[-1]
                continue
            raise RuntimeError('yacc: internal parser error!!!\n')