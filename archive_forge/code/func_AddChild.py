import pickle
def AddChild(self, name, label=None, data=None, isTerminal=0):
    """ Creates a new TreeNode and adds a child to the tree

      **Arguments**

       - name: the name of the new node

       - label: the label of the new node (should be an integer)

       - data: the data to be stored in the new node

       - isTerminal: a toggle to indicate whether or not the new node is
         a terminal (leaf) node.

      **Returns*

        the _TreeNode_ which is constructed

    """
    child = TreeNode(self, name, label, data, level=self.level + 1, isTerminal=isTerminal)
    self.children.append(child)
    return child