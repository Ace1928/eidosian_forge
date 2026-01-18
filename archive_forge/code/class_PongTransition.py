from PySide2.QtWidgets import *
from PySide2.QtCore import *
class PongTransition(QAbstractTransition):

    def eventTest(self, e):
        return e.type() == QEvent.User + 3

    def onTransition(self, e):
        self.p = PingEvent()
        machine.postDelayedEvent(self.p, 500)
        print('ping?')