import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
class TokenInputTransformer(InputTransformer):
    """Wrapper for a token-based input transformer.
    
    func should accept a list of tokens (5-tuples, see tokenize docs), and
    return an iterable which can be passed to tokenize.untokenize().
    """

    def __init__(self, func):
        self.func = func
        self.buf = []
        self.reset_tokenizer()

    def reset_tokenizer(self):
        it = iter(self.buf)
        self.tokenizer = tokenutil.generate_tokens_catch_errors(it.__next__)

    def push(self, line):
        self.buf.append(line + '\n')
        if all((l.isspace() for l in self.buf)):
            return self.reset()
        tokens = []
        stop_at_NL = False
        try:
            for intok in self.tokenizer:
                tokens.append(intok)
                t = intok[0]
                if t == tokenize.NEWLINE or (stop_at_NL and t == tokenize.NL):
                    break
                elif t == tokenize.ERRORTOKEN:
                    stop_at_NL = True
        except TokenError:
            self.reset_tokenizer()
            return None
        return self.output(tokens)

    def output(self, tokens):
        self.buf.clear()
        self.reset_tokenizer()
        return untokenize(self.func(tokens)).rstrip('\n')

    def reset(self):
        l = ''.join(self.buf)
        self.buf.clear()
        self.reset_tokenizer()
        if l:
            return l.rstrip('\n')