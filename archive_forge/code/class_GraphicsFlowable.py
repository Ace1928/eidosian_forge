from io import BytesIO
from reportlab.graphics.shapes import *
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab import rl_config
from reportlab.graphics.renderbase import Renderer, getStateDelta, renderScaledDrawing, STATE_DEFAULTS
from reportlab.platypus import Flowable
class GraphicsFlowable(Flowable):
    """Flowable wrapper around a Pingo drawing"""

    def __init__(self, drawing):
        self.drawing = drawing
        self.width = self.drawing.width
        self.height = self.drawing.height

    def draw(self):
        draw(self.drawing, self.canv, 0, 0)