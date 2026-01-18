def RemoveChild(self, child):
    """Removes a child from our list

          **Arguments**

            - child: a Cluster

        """
    self.children.remove(child)
    self._UpdateLength()