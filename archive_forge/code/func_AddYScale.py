import random
def AddYScale(self, step=1):
    """Add an scale on the y axis."""
    o = self.__offset
    s = self.__scaling
    x = o / 2.0
    dx = self.__offset / 4.0
    self.__dwg.add(self.__dwg.line((x, o), (x, self.__sizey * s + o), stroke='black'))
    for i in range(0, int(self.__sizey) + 1, step):
        self.__dwg.add(self.__dwg.line((x - dx, i * s + o), (x + dx, i * s + o), stroke='black'))