from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import Tree
def _trace_reduce(self, stack, production, remaining_text):
    """
        Print trace output displaying that ``production`` was used to
        reduce ``stack``.

        :rtype: None
        """
    if self._trace > 2:
        rhs = ' '.join(production.rhs())
        print(f'Reduce {production.lhs()!r} <- {rhs}')
    if self._trace == 2:
        self._trace_stack(stack, remaining_text, 'R')
    elif self._trace > 1:
        self._trace_stack(stack, remaining_text)