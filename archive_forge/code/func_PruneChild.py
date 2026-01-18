import pickle
def PruneChild(self, child):
    """ Removes the child node

      **Arguments**

        - child: a TreeNode

    """
    self.children.remove(child)