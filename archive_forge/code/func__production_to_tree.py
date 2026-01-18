from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
def _production_to_tree(self, production):
    """
        :rtype: Tree
        :return: The Tree that is licensed by ``production``.
            In particular, given the production ``[lhs -> elt[1] ... elt[n]]``
            return a tree that has a node ``lhs.symbol``, and
            ``n`` children.  For each nonterminal element
            ``elt[i]`` in the production, the tree token has a
            childless subtree with node value ``elt[i].symbol``; and
            for each terminal element ``elt[j]``, the tree token has
            a leaf token with type ``elt[j]``.

        :param production: The CFG production that licenses the tree
            token that should be returned.
        :type production: Production
        """
    children = []
    for elt in production.rhs():
        if isinstance(elt, Nonterminal):
            children.append(Tree(elt.symbol(), []))
        else:
            children.append(elt)
    return Tree(production.lhs().symbol(), children)