from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
class TrafficLightWidget(QWidget):

    def __init__(self):
        super(TrafficLightWidget, self).__init__()
        vbox = QVBoxLayout(self)
        self.redLight = LightWidget(Qt.red)
        vbox.addWidget(self.redLight)
        self.yellowLight = LightWidget(Qt.yellow)
        vbox.addWidget(self.yellowLight)
        self.greenLight = LightWidget(Qt.green)
        vbox.addWidget(self.greenLight)
        pal = QPalette()
        pal.setColor(QPalette.Background, Qt.black)
        self.setPalette(pal)
        self.setAutoFillBackground(True)