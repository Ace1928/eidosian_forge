import random
from PySide2 import QtCore, QtGui, QtWidgets
class TetrixPiece(object):
    coordsTable = (((0, 0), (0, 0), (0, 0), (0, 0)), ((0, -1), (0, 0), (-1, 0), (-1, 1)), ((0, -1), (0, 0), (1, 0), (1, 1)), ((0, -1), (0, 0), (0, 1), (0, 2)), ((-1, 0), (0, 0), (1, 0), (0, 1)), ((0, 0), (1, 0), (0, 1), (1, 1)), ((-1, -1), (0, -1), (0, 0), (0, 1)), ((1, -1), (0, -1), (0, 0), (0, 1)))

    def __init__(self):
        self.coords = [[0, 0] for _ in range(4)]
        self.pieceShape = NoShape
        self.setShape(NoShape)

    def shape(self):
        return self.pieceShape

    def setShape(self, shape):
        table = TetrixPiece.coordsTable[shape]
        for i in range(4):
            for j in range(2):
                self.coords[i][j] = table[i][j]
        self.pieceShape = shape

    def setRandomShape(self):
        self.setShape(random.randint(1, 7))

    def x(self, index):
        return self.coords[index][0]

    def y(self, index):
        return self.coords[index][1]

    def setX(self, index, x):
        self.coords[index][0] = x

    def setY(self, index, y):
        self.coords[index][1] = y

    def minX(self):
        m = self.coords[0][0]
        for i in range(4):
            m = min(m, self.coords[i][0])
        return m

    def maxX(self):
        m = self.coords[0][0]
        for i in range(4):
            m = max(m, self.coords[i][0])
        return m

    def minY(self):
        m = self.coords[0][1]
        for i in range(4):
            m = min(m, self.coords[i][1])
        return m

    def maxY(self):
        m = self.coords[0][1]
        for i in range(4):
            m = max(m, self.coords[i][1])
        return m

    def rotatedLeft(self):
        if self.pieceShape == SquareShape:
            return self
        result = TetrixPiece()
        result.pieceShape = self.pieceShape
        for i in range(4):
            result.setX(i, self.y(i))
            result.setY(i, -self.x(i))
        return result

    def rotatedRight(self):
        if self.pieceShape == SquareShape:
            return self
        result = TetrixPiece()
        result.pieceShape = self.pieceShape
        for i in range(4):
            result.setX(i, -self.y(i))
            result.setY(i, self.x(i))
        return result