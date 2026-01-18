from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
class paraMapper(MapNode):

    def translate(self, nodetuple, controller, context, overrides):
        tagname, attdict, content, extra = nodetuple
        stylename = tagname + '.defaultStyle'
        stylename = attdict.get('style', stylename)
        style = context[stylename]
        mystyle = SimpleStyle(name='rml2pdf internal style', parent=style)
        mystyle.addAttributes(attdict)
        bulletText = attdict.get('bulletText', None)
        result = None
        if not bulletText and len(content) == 1:
            text = content[0]
            if isinstance(text, str) and '&' not in text:
                result = FastPara(mystyle, text)
        if result is None:
            result = Para(mystyle, content, context=context, bulletText=bulletText)
        return result