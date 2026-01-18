from PySide2.QtWidgets import (QWidget)
from PySide2.QtGui import (QPainter)
from PySide2.QtCore import Signal
class StarEditor(QWidget):
    """ The custome editor for editing StarRatings. """
    editingFinished = Signal()

    def __init__(self, parent=None):
        """ Initialize the editor object, making sure we can watch mouse
            events.
        """
        super(StarEditor, self).__init__(parent)
        self.setMouseTracking(True)
        self.setAutoFillBackground(True)

    def sizeHint(self):
        """ Tell the caller how big we are. """
        return self.starRating.sizeHint()

    def paintEvent(self, event):
        """ Paint the editor, offloading the work to the StarRating class. """
        painter = QPainter(self)
        self.starRating.paint(painter, self.rect(), self.palette(), isEditable=True)

    def mouseMoveEvent(self, event):
        """ As the mouse moves inside the editor, track the position and
            update the editor to display as many stars as necessary.
        """
        star = self.starAtPosition(event.x())
        if star != self.starRating.starCount and star != -1:
            self.starRating.starCount = star
            self.update()

    def mouseReleaseEvent(self, event):
        """ Once the user has clicked his/her chosen star rating, tell the
            delegate we're done editing.
        """
        self.editingFinished.emit()

    def starAtPosition(self, x):
        """ Calculate which star the user's mouse cursor is currently
            hovering over.
        """
        star = x / (self.starRating.sizeHint().width() / self.starRating.maxStarCount) + 1
        if star <= 0 or star > self.starRating.maxStarCount:
            return -1
        return star