from PySide2.QtWidgets import *
from PySide2.QtCore import *
class PingTransition(QAbstractTransition):

    def eventTest(self, e):
        return e.type() == QEvent.User + 2

    def onTransition(self, e):
        self.p = PongEvent()
        machine.postDelayedEvent(self.p, 500)
        print('pong!')