from PySide2 import QtCore, QtGui, QtWidgets, QtXml
class XbelTree(QtWidgets.QTreeWidget):

    def __init__(self, parent=None):
        super(XbelTree, self).__init__(parent)
        self.header().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.setHeaderLabels(('Title', 'Location'))
        self.domDocument = QtXml.QDomDocument()
        self.domElementForItem = {}
        self.folderIcon = QtGui.QIcon()
        self.bookmarkIcon = QtGui.QIcon()
        self.folderIcon.addPixmap(self.style().standardPixmap(QtWidgets.QStyle.SP_DirClosedIcon), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.folderIcon.addPixmap(self.style().standardPixmap(QtWidgets.QStyle.SP_DirOpenIcon), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.bookmarkIcon.addPixmap(self.style().standardPixmap(QtWidgets.QStyle.SP_FileIcon))

    def read(self, device):
        ok, errorStr, errorLine, errorColumn = self.domDocument.setContent(device, True)
        if not ok:
            QtWidgets.QMessageBox.information(self.window(), 'DOM Bookmarks', 'Parse error at line %d, column %d:\n%s' % (errorLine, errorColumn, errorStr))
            return False
        root = self.domDocument.documentElement()
        if root.tagName() != 'xbel':
            QtWidgets.QMessageBox.information(self.window(), 'DOM Bookmarks', 'The file is not an XBEL file.')
            return False
        elif root.hasAttribute('version') and root.attribute('version') != '1.0':
            QtWidgets.QMessageBox.information(self.window(), 'DOM Bookmarks', 'The file is not an XBEL version 1.0 file.')
            return False
        self.clear()
        try:
            self.itemChanged.disconnect(self.updateDomElement)
        except:
            pass
        child = root.firstChildElement('folder')
        while not child.isNull():
            self.parseFolderElement(child)
            child = child.nextSiblingElement('folder')
        self.itemChanged.connect(self.updateDomElement)
        return True

    def write(self, device):
        indentSize = 4
        out = QtCore.QTextStream(device)
        self.domDocument.save(out, indentSize)
        return True

    def updateDomElement(self, item, column):
        element = self.domElementForItem.get(id(item))
        if not element.isNull():
            if column == 0:
                oldTitleElement = element.firstChildElement('title')
                newTitleElement = self.domDocument.createElement('title')
                newTitleText = self.domDocument.createTextNode(item.text(0))
                newTitleElement.appendChild(newTitleText)
                element.replaceChild(newTitleElement, oldTitleElement)
            elif element.tagName() == 'bookmark':
                element.setAttribute('href', item.text(1))

    def parseFolderElement(self, element, parentItem=None):
        item = self.createItem(element, parentItem)
        title = element.firstChildElement('title').text()
        if not title:
            title = 'Folder'
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        item.setIcon(0, self.folderIcon)
        item.setText(0, title)
        folded = element.attribute('folded') != 'no'
        self.setItemExpanded(item, not folded)
        child = element.firstChildElement()
        while not child.isNull():
            if child.tagName() == 'folder':
                self.parseFolderElement(child, item)
            elif child.tagName() == 'bookmark':
                childItem = self.createItem(child, item)
                title = child.firstChildElement('title').text()
                if not title:
                    title = 'Folder'
                childItem.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
                childItem.setIcon(0, self.bookmarkIcon)
                childItem.setText(0, title)
                childItem.setText(1, child.attribute('href'))
            elif child.tagName() == 'separator':
                childItem = self.createItem(child, item)
                childItem.setFlags(item.flags() & ~(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable))
                childItem.setText(0, 30 * '·')
            child = child.nextSiblingElement()

    def createItem(self, element, parentItem=None):
        item = QtWidgets.QTreeWidgetItem()
        if parentItem is not None:
            item = QtWidgets.QTreeWidgetItem(parentItem)
        else:
            item = QtWidgets.QTreeWidgetItem(self)
        self.domElementForItem[id(item)] = element
        return item