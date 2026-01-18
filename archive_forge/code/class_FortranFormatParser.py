import re
import numpy as np
class FortranFormatParser:
    """Parser for Fortran format strings. The parse method returns a *Format
    instance.

    Notes
    -----
    Only ExpFormat (exponential format for floating values) and IntFormat
    (integer format) for now.
    """

    def __init__(self):
        self.tokenizer = Tokenizer()

    def parse(self, s):
        self.tokenizer.input(s)
        tokens = []
        try:
            while True:
                t = self.tokenizer.next_token()
                if t is None:
                    break
                else:
                    tokens.append(t)
            return self._parse_format(tokens)
        except SyntaxError as e:
            raise BadFortranFormat(str(e)) from e

    def _get_min(self, tokens):
        next = tokens.pop(0)
        if not next.type == 'DOT':
            raise SyntaxError()
        next = tokens.pop(0)
        return next.value

    def _expect(self, token, tp):
        if not token.type == tp:
            raise SyntaxError()

    def _parse_format(self, tokens):
        if not tokens[0].type == 'LPAR':
            raise SyntaxError("Expected left parenthesis at position %d (got '%s')" % (0, tokens[0].value))
        elif not tokens[-1].type == 'RPAR':
            raise SyntaxError("Expected right parenthesis at position %d (got '%s')" % (len(tokens), tokens[-1].value))
        tokens = tokens[1:-1]
        types = [t.type for t in tokens]
        if types[0] == 'INT':
            repeat = int(tokens.pop(0).value)
        else:
            repeat = None
        next = tokens.pop(0)
        if next.type == 'INT_ID':
            next = self._next(tokens, 'INT')
            width = int(next.value)
            if tokens:
                min = int(self._get_min(tokens))
            else:
                min = None
            return IntFormat(width, min, repeat)
        elif next.type == 'EXP_ID':
            next = self._next(tokens, 'INT')
            width = int(next.value)
            next = self._next(tokens, 'DOT')
            next = self._next(tokens, 'INT')
            significand = int(next.value)
            if tokens:
                next = self._next(tokens, 'EXP_ID')
                next = self._next(tokens, 'INT')
                min = int(next.value)
            else:
                min = None
            return ExpFormat(width, significand, min, repeat)
        else:
            raise SyntaxError('Invalid formatter type %s' % next.value)

    def _next(self, tokens, tp):
        if not len(tokens) > 0:
            raise SyntaxError()
        next = tokens.pop(0)
        self._expect(next, tp)
        return next