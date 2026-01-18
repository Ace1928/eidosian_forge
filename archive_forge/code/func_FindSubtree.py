def FindSubtree(self, index):
    """ finds and returns the subtree with a particular index
        """
    res = None
    if index == self.index:
        res = self
    else:
        for child in self.children:
            res = child.FindSubtree(index)
            if res:
                break
    return res