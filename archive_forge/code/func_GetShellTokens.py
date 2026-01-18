from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
def GetShellTokens(s):
    """Returns the list of ShellTokens in s.

  Args:
    s: The string to parse for shell tokens.

  Returns:
    The list of ShellTokens in s.
  """
    tokens = []
    i = 0
    while True:
        token = GetShellToken(i, s)
        if not token:
            break
        i = token.end
        tokens.append(token)
        if token.lex == ShellTokenType.REDIRECTION:
            token = GetShellToken(i, s)
            if not token:
                break
            i = token.end
            token.lex = ShellTokenType.FILE
            tokens.append(token)
    return tokens