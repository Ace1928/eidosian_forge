from PySide2.QtCore import (Signal, QDataStream, QMutex, QMutexLocker,
from PySide2.QtGui import QIntValidator
from PySide2.QtWidgets import (QApplication, QDialogButtonBox, QGridLayout,
from PySide2.QtNetwork import (QAbstractSocket, QHostAddress, QNetworkInterface,
class FortuneThread(QThread):
    newFortune = Signal(str)
    error = Signal(int, str)

    def __init__(self, parent=None):
        super(FortuneThread, self).__init__(parent)
        self.quit = False
        self.hostName = ''
        self.cond = QWaitCondition()
        self.mutex = QMutex()
        self.port = 0

    def __del__(self):
        self.mutex.lock()
        self.quit = True
        self.cond.wakeOne()
        self.mutex.unlock()
        self.wait()

    def requestNewFortune(self, hostname, port):
        locker = QMutexLocker(self.mutex)
        self.hostName = hostname
        self.port = port
        if not self.isRunning():
            self.start()
        else:
            self.cond.wakeOne()

    def run(self):
        self.mutex.lock()
        serverName = self.hostName
        serverPort = self.port
        self.mutex.unlock()
        while not self.quit:
            Timeout = 5 * 1000
            socket = QTcpSocket()
            socket.connectToHost(serverName, serverPort)
            if not socket.waitForConnected(Timeout):
                self.error.emit(socket.error(), socket.errorString())
                return
            while socket.bytesAvailable() < 2:
                if not socket.waitForReadyRead(Timeout):
                    self.error.emit(socket.error(), socket.errorString())
                    return
            instr = QDataStream(socket)
            instr.setVersion(QDataStream.Qt_4_0)
            blockSize = instr.readUInt16()
            while socket.bytesAvailable() < blockSize:
                if not socket.waitForReadyRead(Timeout):
                    self.error.emit(socket.error(), socket.errorString())
                    return
            self.mutex.lock()
            fortune = instr.readQString()
            self.newFortune.emit(fortune)
            self.cond.wait(self.mutex)
            serverName = self.hostName
            serverPort = self.port
            self.mutex.unlock()