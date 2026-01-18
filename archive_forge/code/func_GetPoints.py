def GetPoints(self):
    if self._points is not None:
        return self._points
    else:
        self._GenPoints()
        return self._points