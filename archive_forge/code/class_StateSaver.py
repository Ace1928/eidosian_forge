from rdkit.sping.colors import *
class StateSaver:
    """This is a little utility class for saving and restoring the
          default drawing parameters of a canvas.  To use it, add a line
          like this before changing any of the parameters:

                  saver = StateSaver(myCanvas)

          then, when "saver" goes out of scope, it will automagically
          restore the drawing parameters of myCanvas."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.defaultLineColor = canvas.defaultLineColor
        self.defaultFillColor = canvas.defaultFillColor
        self.defaultLineWidth = canvas.defaultLineWidth
        self.defaultFont = canvas.defaultFont

    def __del__(self):
        self.canvas.defaultLineColor = self.defaultLineColor
        self.canvas.defaultFillColor = self.defaultFillColor
        self.canvas.defaultLineWidth = self.defaultLineWidth
        self.canvas.defaultFont = self.defaultFont