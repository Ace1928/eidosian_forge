from nltk.probability import ProbabilisticMixIn
from nltk.tree.parented import MultiParentedTree, ParentedTree
from nltk.tree.tree import Tree
class ImmutableMultiParentedTree(ImmutableTree, MultiParentedTree):
    pass