from math import (cos, sin, pi)
from PySide2.QtGui import (QPainter, QPolygonF)
from PySide2.QtCore import (QPointF, QSize, Qt)
class StarRating(object):
    """ Handle the actual painting of the stars themselves. """

    def __init__(self, starCount=1, maxStarCount=5):
        self.starCount = starCount
        self.maxStarCount = maxStarCount
        self.starPolygon = QPolygonF()
        self.starPolygon.append(QPointF(1.0, 0.5))
        for i in range(1, 5):
            self.starPolygon.append(QPointF(0.5 + 0.5 * cos(0.8 * i * pi), 0.5 + 0.5 * sin(0.8 * i * pi)))
        self.diamondPolygon = QPolygonF()
        diamondPoints = [QPointF(0.4, 0.5), QPointF(0.5, 0.4), QPointF(0.6, 0.5), QPointF(0.5, 0.6), QPointF(0.4, 0.5)]
        for point in diamondPoints:
            self.diamondPolygon.append(point)

    def sizeHint(self):
        """ Tell the caller how big we are. """
        return PAINTING_SCALE_FACTOR * QSize(self.maxStarCount, 1)

    def paint(self, painter, rect, palette, isEditable=False):
        """ Paint the stars (and/or diamonds if we're in editing mode). """
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        if isEditable:
            painter.setBrush(palette.highlight())
        else:
            painter.setBrush(palette.windowText())
        yOffset = (rect.height() - PAINTING_SCALE_FACTOR) / 2
        painter.translate(rect.x(), rect.y() + yOffset)
        painter.scale(PAINTING_SCALE_FACTOR, PAINTING_SCALE_FACTOR)
        for i in range(self.maxStarCount):
            if i < self.starCount:
                painter.drawPolygon(self.starPolygon, Qt.WindingFill)
            elif isEditable:
                painter.drawPolygon(self.diamondPolygon, Qt.WindingFill)
            painter.translate(1.0, 0.0)
        painter.restore()