import sys, collections, pyrect
def _setupRectProperties(self):

    def _onRead(attrName):
        r = self._getWindowRect()
        self._rect._left = r.left
        self._rect._top = r.top
        self._rect._width = r.right - r.left
        self._rect._height = r.bottom - r.top

    def _onChange(oldBox, newBox):
        self.moveTo(newBox.left, newBox.top)
        self.resizeTo(newBox.width, newBox.height)
    r = self._getWindowRect()
    self._rect = pyrect.Rect(r.left, r.top, r.right - r.left, r.bottom - r.top, onChange=_onChange, onRead=_onRead)