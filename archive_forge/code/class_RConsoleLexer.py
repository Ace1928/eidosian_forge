import re
from pygments.lexer import Lexer, RegexLexer, include, words, do_insertions
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class RConsoleLexer(Lexer):
    """
    For R console transcripts or R CMD BATCH output files.
    """
    name = 'RConsole'
    aliases = ['rconsole', 'rout']
    filenames = ['*.Rout']

    def get_tokens_unprocessed(self, text):
        slexer = SLexer(**self.options)
        current_code_block = ''
        insertions = []
        for match in line_re.finditer(text):
            line = match.group()
            if line.startswith('>') or line.startswith('+'):
                insertions.append((len(current_code_block), [(0, Generic.Prompt, line[:2])]))
                current_code_block += line[2:]
            else:
                if current_code_block:
                    for item in do_insertions(insertions, slexer.get_tokens_unprocessed(current_code_block)):
                        yield item
                    current_code_block = ''
                    insertions = []
                yield (match.start(), Generic.Output, line)
        if current_code_block:
            for item in do_insertions(insertions, slexer.get_tokens_unprocessed(current_code_block)):
                yield item