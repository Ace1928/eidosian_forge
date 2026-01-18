def _GenPoints(self):
    """ Generates the _Points_ and _PointsPositions_ lists

         *intended for internal use*

        """
    if len(self) == 1:
        self._points = [self]
        self._pointsPositions = [self.GetPosition()]
        return self._points
    else:
        res = []
        children = self.GetChildren()
        children.sort(key=lambda x: len(x), reverse=True)
        for child in children:
            res += child.GetPoints()
        self._points = res
        self._pointsPositions = [x.GetPosition() for x in res]