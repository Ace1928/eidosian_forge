from PySide2 import QtCore, QtGui, QtWidgets
class FileListModel(QtCore.QAbstractListModel):
    numberPopulated = QtCore.Signal(int)

    def __init__(self, parent=None):
        super(FileListModel, self).__init__(parent)
        self.fileCount = 0
        self.fileList = []

    def rowCount(self, parent=QtCore.QModelIndex()):
        return self.fileCount

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        if index.row() >= len(self.fileList) or index.row() < 0:
            return None
        if role == QtCore.Qt.DisplayRole:
            return self.fileList[index.row()]
        if role == QtCore.Qt.BackgroundRole:
            batch = index.row() // 100 % 2
            if batch == 0:
                return QtWidgets.qApp.palette().base()
            return QtWidgets.qApp.palette().alternateBase()
        return None

    def canFetchMore(self, index):
        return self.fileCount < len(self.fileList)

    def fetchMore(self, index):
        remainder = len(self.fileList) - self.fileCount
        itemsToFetch = min(100, remainder)
        self.beginInsertRows(QtCore.QModelIndex(), self.fileCount, self.fileCount + itemsToFetch)
        self.fileCount += itemsToFetch
        self.endInsertRows()
        self.numberPopulated.emit(itemsToFetch)

    def setDirPath(self, path):
        dir = QtCore.QDir(path)
        self.beginResetModel()
        self.fileList = list(dir.entryList())
        self.fileCount = 0
        self.endResetModel()