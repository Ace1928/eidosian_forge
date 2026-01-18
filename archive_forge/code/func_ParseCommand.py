from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.interactive import lexer
import six
def ParseCommand(self, line):
    """Parses the next command from line and returns a list of ArgTokens.

    The parse stops at the first token that is not an ARG or FLAG. That token is
    not consumed. The caller can examine the return value to determine the
    parts of the line that were ignored and the remainder of the line that was
    not lexed/parsed yet.

    Args:
      line: a string containing the current command line

    Returns:
      A list of ArgTokens.
    """
    self.tokens = lexer.GetShellTokens(line)
    self.cmd = self.root
    self.positionals_seen = 0
    self.args = []
    unknown = False
    while self.tokens:
        token = self.tokens.pop(0)
        value = token.UnquotedValue()
        if token.lex == lexer.ShellTokenType.TERMINATOR:
            unknown = False
            self.cmd = self.root
            self.args.append(ArgToken(value, ArgTokenType.SPECIAL, self.cmd, token.start, token.end))
        elif token.lex == lexer.ShellTokenType.FLAG:
            self.ParseFlag(token, value)
        elif token.lex == lexer.ShellTokenType.ARG and (not unknown):
            if value in self.cmd[LOOKUP_COMMANDS]:
                self.cmd = self.cmd[LOOKUP_COMMANDS][value]
                if self.cmd[LOOKUP_IS_GROUP]:
                    token_type = ArgTokenType.GROUP
                elif LOOKUP_IS_SPECIAL in self.cmd:
                    token_type = ArgTokenType.SPECIAL
                    self.cmd = self.root
                else:
                    token_type = ArgTokenType.COMMAND
                self.args.append(ArgToken(value, token_type, self.cmd, token.start, token.end))
            elif self.cmd == self.root and '=' in value:
                token_type = ArgTokenType.SPECIAL
                self.cmd = self.root
                self.args.append(ArgToken(value, token_type, self.cmd, token.start, token.end))
            elif self.positionals_seen < len(self.cmd[LOOKUP_POSITIONALS]):
                positional = self.cmd[LOOKUP_POSITIONALS][self.positionals_seen]
                self.args.append(ArgToken(value, ArgTokenType.POSITIONAL, positional, token.start, token.end))
                if positional[LOOKUP_NARGS] not in ('*', '+'):
                    self.positionals_seen += 1
            elif not value:
                break
            else:
                unknown = True
                if self.cmd == self.root:
                    token_type = ArgTokenType.PREFIX
                else:
                    token_type = ArgTokenType.UNKNOWN
                self.args.append(ArgToken(value, token_type, self.cmd, token.start, token.end))
        else:
            unknown = True
            self.args.append(ArgToken(value, ArgTokenType.UNKNOWN, self.cmd, token.start, token.end))
    return self.args