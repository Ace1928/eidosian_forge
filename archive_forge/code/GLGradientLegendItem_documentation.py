from ... import functions as fn
from ...colormap import ColorMap
from ...Qt import QtCore, QtGui
from ..GLGraphicsItem import GLGraphicsItem

        Arguments:
            pos: position of the colorbar on the screen, from the top left corner, in pixels
            size: size of the colorbar without the text, in pixels
            gradient: a pg.ColorMap used to color the colorbar
            labels: a dict of "text":value to display next to the colorbar.
                The value corresponds to a position in the gradient from 0 to 1.
            fontColor: sets the color of the texts. Accepts any single argument accepted by
                :func:`~pyqtgraph.mkColor`
            #Todo:
                size as percentage
                legend title
        