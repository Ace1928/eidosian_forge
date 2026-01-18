import copy
from math import *
from qt import *
from qtcanvas import *
from rdkit.sping import pid
class QCanvasRotText(QCanvasText):
    """ used to draw (UGLY) rotated text

  """

    def __init__(self, txt, canvas, angle=0):
        QCanvasText.__init__(self, txt, canvas)
        self._angle = angle

    def draw(self, qP):
        qP.save()
        x = self.x()
        y = self.y()
        theta = -self._angle
        qP.rotate(theta)
        qP.translate(-x, -y)
        thetaR = theta * pi / 180.0
        newX = cos(-thetaR) * x - sin(-thetaR) * y
        newY = sin(-thetaR) * x + cos(-thetaR) * y
        qP.translate(newX, newY)
        QCanvasText.draw(self, qP)
        qP.restore()