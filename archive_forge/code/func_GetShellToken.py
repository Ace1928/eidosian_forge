from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
def GetShellToken(i, s):
    """Returns the next shell token at s[i:].

  Args:
    i: The index of the next character in s.
    s: The string to parse for shell tokens.

  Returns:
   A ShellToken, None if there are no more token in s.
  """
    index = i
    while True:
        if i >= len(s):
            if i > index:
                return ShellToken('', ShellTokenType.ARG, i - 1, i)
            return None
        c = s[i]
        if not c.isspace():
            break
        i += 1
    index = i
    if len(s) - 1 == i and s[i] == '\\':
        i += 1
        return ShellToken(s[index:i], ShellTokenType.TRAILING_BACKSLASH, index, i)
    if c in SHELL_REDIRECTION_CHARS or (c.isdigit() and i + 1 < len(s) and (s[i + 1] in SHELL_REDIRECTION_CHARS)):
        index = i
        if s[i].isdigit():
            i += 1
        if i < len(s) and s[i] in SHELL_REDIRECTION_CHARS:
            i += 1
            while i < len(s) and s[i] in SHELL_REDIRECTION_CHARS:
                i += 1
            if i < len(s) - 1 and s[i] == '&' and s[i + 1].isdigit():
                i += 2
                lex = ShellTokenType.IO
            else:
                lex = ShellTokenType.REDIRECTION
            return ShellToken(s[index:i], lex, start=index, end=i)
        i = index
    if c in SHELL_TERMINATOR_CHARS:
        i += 1
        return ShellToken(s[index:i], ShellTokenType.TERMINATOR, start=index, end=i)
    quote = None
    while i < len(s):
        c = s[i]
        if c == quote:
            quote = None
        elif quote is None:
            if c in SHELL_QUOTE_CHARS:
                quote = c
            elif c == SHELL_ESCAPE_CHAR:
                if i + 1 < len(s):
                    i += 1
                else:
                    break
            elif c.isspace():
                break
            elif c in SHELL_TERMINATOR_CHARS:
                break
        i += 1
    lex = ShellTokenType.FLAG if s[index] == '-' else ShellTokenType.ARG
    return ShellToken(s[index:i], lex, start=index, end=i)