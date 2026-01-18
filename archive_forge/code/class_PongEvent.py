from PySide2.QtWidgets import *
from PySide2.QtCore import *
class PongEvent(QEvent):

    def __init__(self):
        super(PongEvent, self).__init__(QEvent.Type(QEvent.User + 3))