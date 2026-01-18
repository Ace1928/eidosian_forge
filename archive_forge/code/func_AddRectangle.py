import random
def AddRectangle(self, x, y, dx, dy, fill, stroke='black', label=None):
    """Draw a rectangle, dx and dy must be >= 0."""
    s = self.__scaling
    o = self.__offset
    corner = (x * s + o, (self.__sizey - y - dy) * s + o)
    size = (dx * s - 1, dy * s - 1)
    self.__dwg.add(self.__dwg.rect(insert=corner, size=size, fill=fill, stroke=stroke))
    self.AddText(x + 0.5 * dx, y + 0.5 * dy, label)