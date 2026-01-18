import warnings
import json
import random
from .base import Renderer
from ..exporter import Exporter
class VegaHTML(object):

    def __init__(self, renderer):
        self.specification = dict(width=renderer.figwidth, height=renderer.figheight, data=renderer.data, scales=renderer.scales, axes=renderer.axes, marks=renderer.marks)

    def html(self):
        """Build the HTML representation for IPython."""
        id = random.randint(0, 2 ** 16)
        html = '<div id="vis%d"></div>' % id
        html += '<script>\n'
        html += VEGA_TEMPLATE % (json.dumps(self.specification), id)
        html += '</script>\n'
        return html

    def _repr_html_(self):
        return self.html()