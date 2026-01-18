import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, Whitespace, \
from pygments.util import shebang_matches
class SlurmBashLexer(BashLexer):
    """
    Lexer for (ba|k|z|)sh Slurm scripts.

    .. versionadded:: 2.4
    """
    name = 'Slurm'
    aliases = ['slurm', 'sbatch']
    filenames = ['*.sl']
    mimetypes = []
    EXTRA_KEYWORDS = {'srun'}

    def get_tokens_unprocessed(self, text):
        for index, token, value in BashLexer.get_tokens_unprocessed(self, text):
            if token is Text and value in self.EXTRA_KEYWORDS:
                yield (index, Name.Builtin, value)
            elif token is Comment.Single and 'SBATCH' in value:
                yield (index, Keyword.Pseudo, value)
            else:
                yield (index, token, value)