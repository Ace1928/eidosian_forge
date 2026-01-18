import random
from PySide2.QtCore import (Signal, QByteArray, QDataStream, QIODevice,
from PySide2.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
from PySide2.QtNetwork import (QHostAddress, QNetworkInterface, QTcpServer,
class FortuneServer(QTcpServer):
    fortunes = ("You've been leading a dog's life. Stay off the furniture.", "You've got to think about tomorrow.", 'You will be surprised by a loud noise.', 'You will feel hungry again in another hour.', 'You might have mail.', 'You cannot kill time without injuring eternity.', 'Computers are not intelligent. They only think they are.')

    def incomingConnection(self, socketDescriptor):
        fortune = self.fortunes[random.randint(0, len(self.fortunes) - 1)]
        thread = FortuneThread(socketDescriptor, fortune, self)
        thread.finished.connect(thread.deleteLater)
        thread.start()