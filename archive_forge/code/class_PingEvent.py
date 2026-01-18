from PySide2.QtWidgets import *
from PySide2.QtCore import *
class PingEvent(QEvent):

    def __init__(self):
        super(PingEvent, self).__init__(QEvent.Type(QEvent.User + 2))