import pickle
def AddChildNode(self, node):
    """ Adds a TreeNode to the local list of children

     **Arguments**

       - node: the node to be added

     **Note**

       the level of the node (used in printing) is set as well

    """
    node.SetLevel(self.level + 1)
    self.children.append(node)