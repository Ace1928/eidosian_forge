import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(value='continue')
class _ContinueChecks(SyntaxRule):
    message = "'continue' not properly in loop"
    message_in_finally = "'continue' not supported inside 'finally' clause"

    def is_issue(self, leaf):
        in_loop = False
        for block in self._normalizer.context.blocks:
            if block.type in ('for_stmt', 'while_stmt'):
                in_loop = True
            if block.type == 'try_stmt':
                last_block = block.children[-3]
                if last_block == 'finally' and leaf.start_pos > last_block.start_pos and (self._normalizer.version < (3, 8)):
                    self.add_issue(leaf, message=self.message_in_finally)
                    return False
        if not in_loop:
            return True