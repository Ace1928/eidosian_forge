from PySide2.QtWidgets import *
from PySide2.QtCore import *
class FactorialLoopTransition(QSignalTransition):

    def __init__(self, fact):
        super(FactorialLoopTransition, self).__init__(fact, SIGNAL('xChanged(int)'))
        self.fact = fact

    def eventTest(self, e):
        if not super(FactorialLoopTransition, self).eventTest(e):
            return False
        return e.arguments()[0] > 1

    def onTransition(self, e):
        x = e.arguments()[0]
        fac = self.fact.fac
        self.fact.fac = x * fac
        self.fact.x = x - 1