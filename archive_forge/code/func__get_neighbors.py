import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
def _get_neighbors(self, tree):
    """Get all neighbor trees of the given tree (PRIVATE).

        Currently only for binary rooted trees.
        """
    parents = {}
    for clade in tree.find_clades():
        if clade != tree.root:
            node_path = tree.get_path(clade)
            if len(node_path) == 1:
                parents[clade] = tree.root
            else:
                parents[clade] = node_path[-2]
    neighbors = []
    root_childs = []
    for clade in tree.get_nonterminals(order='level'):
        if clade == tree.root:
            left = clade.clades[0]
            right = clade.clades[1]
            root_childs.append(left)
            root_childs.append(right)
            if not left.is_terminal() and (not right.is_terminal()):
                left_right = left.clades[1]
                right_left = right.clades[0]
                right_right = right.clades[1]
                del left.clades[1]
                del right.clades[1]
                left.clades.append(right_right)
                right.clades.append(left_right)
                temp_tree = copy.deepcopy(tree)
                neighbors.append(temp_tree)
                del left.clades[1]
                del right.clades[0]
                left.clades.append(right_left)
                right.clades.append(right_right)
                temp_tree = copy.deepcopy(tree)
                neighbors.append(temp_tree)
                del left.clades[1]
                del right.clades[0]
                left.clades.append(left_right)
                right.clades.insert(0, right_left)
        elif clade in root_childs:
            continue
        else:
            left = clade.clades[0]
            right = clade.clades[1]
            parent = parents[clade]
            if clade == parent.clades[0]:
                sister = parent.clades[1]
                del parent.clades[1]
                del clade.clades[1]
                parent.clades.append(right)
                clade.clades.append(sister)
                temp_tree = copy.deepcopy(tree)
                neighbors.append(temp_tree)
                del parent.clades[1]
                del clade.clades[0]
                parent.clades.append(left)
                clade.clades.append(right)
                temp_tree = copy.deepcopy(tree)
                neighbors.append(temp_tree)
                del parent.clades[1]
                del clade.clades[0]
                parent.clades.append(sister)
                clade.clades.insert(0, left)
            else:
                sister = parent.clades[0]
                del parent.clades[0]
                del clade.clades[1]
                parent.clades.insert(0, right)
                clade.clades.append(sister)
                temp_tree = copy.deepcopy(tree)
                neighbors.append(temp_tree)
                del parent.clades[0]
                del clade.clades[0]
                parent.clades.insert(0, left)
                clade.clades.append(right)
                temp_tree = copy.deepcopy(tree)
                neighbors.append(temp_tree)
                del parent.clades[0]
                del clade.clades[0]
                parent.clades.insert(0, sister)
                clade.clades.insert(0, left)
    return neighbors