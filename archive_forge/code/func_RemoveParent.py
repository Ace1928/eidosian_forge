import sys
def RemoveParent(self, parent, notify=True):
    """
        >>> p1 = VLibNode()
        >>> c1 = VLibNode()
        >>> p1.AddChild(c1)
        >>> len(c1.GetParents())
        1
        >>> len(p1.GetChildren())
        1
        >>> c1.RemoveParent(p1)
        >>> len(c1.GetParents())
        0
        >>> len(p1.GetChildren())
        0
        """
    self._parents.remove(parent)
    if notify:
        parent.RemoveChild(self, notify=False)