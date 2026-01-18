from nltk.probability import ProbabilisticMixIn
from nltk.tree.parented import MultiParentedTree, ParentedTree
from nltk.tree.tree import Tree
class ImmutableParentedTree(ImmutableTree, ParentedTree):
    pass