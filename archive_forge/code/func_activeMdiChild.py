from PySide2.QtCore import (QFile, QFileInfo, QPoint, QSettings, QSignalMapper,
from PySide2.QtGui import QIcon, QKeySequence
from PySide2.QtWidgets import (QAction, QApplication, QFileDialog, QMainWindow,
import mdi_rc
def activeMdiChild(self):
    activeSubWindow = self.mdiArea.activeSubWindow()
    if activeSubWindow:
        return activeSubWindow.widget()
    return None