import itertools as it
from abc import ABCMeta, abstractmethod
from nltk.tbl.feature import Feature
from nltk.tbl.rule import Rule
class BrillTemplateI(metaclass=ABCMeta):
    """
    An interface for generating lists of transformational rules that
    apply at given sentence positions.  ``BrillTemplateI`` is used by
    ``Brill`` training algorithms to generate candidate rules.
    """

    @abstractmethod
    def applicable_rules(self, tokens, i, correctTag):
        """
        Return a list of the transformational rules that would correct
        the ``i``-th subtoken's tag in the given token.  In particular,
        return a list of zero or more rules that would change
        ``tokens[i][1]`` to ``correctTag``, if applied to ``token[i]``.

        If the ``i``-th token already has the correct tag (i.e., if
        ``tagged_tokens[i][1] == correctTag``), then
        ``applicable_rules()`` should return the empty list.

        :param tokens: The tagged tokens being tagged.
        :type tokens: list(tuple)
        :param i: The index of the token whose tag should be corrected.
        :type i: int
        :param correctTag: The correct tag for the ``i``-th token.
        :type correctTag: any
        :rtype: list(BrillRule)
        """

    @abstractmethod
    def get_neighborhood(self, token, index):
        """
        Returns the set of indices *i* such that
        ``applicable_rules(token, i, ...)`` depends on the value of
        the *index*th token of *token*.

        This method is used by the "fast" Brill tagger trainer.

        :param token: The tokens being tagged.
        :type token: list(tuple)
        :param index: The index whose neighborhood should be returned.
        :type index: int
        :rtype: set
        """