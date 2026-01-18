from PySide2.QtCore import (Signal, QDataStream, QMutex, QMutexLocker,
from PySide2.QtGui import QIntValidator
from PySide2.QtWidgets import (QApplication, QDialogButtonBox, QGridLayout,
from PySide2.QtNetwork import (QAbstractSocket, QHostAddress, QNetworkInterface,
PySide2 port of the network/blockingfortunclient example from Qt v5.x, originating from PyQt