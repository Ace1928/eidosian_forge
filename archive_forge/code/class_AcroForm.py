from reportlab.pdfbase.pdfdoc import PDFString, PDFStream, PDFDictionary, PDFName, PDFObject
from reportlab.lib.colors import obj_R_G_B
from reportlab.pdfbase.pdfpattern import PDFPattern, PDFPatternIf
from reportlab.rl_config import register_reset
class AcroForm(PDFObject):

    def __init__(self):
        self.fields = []

    def textField(self, canvas, title, xmin, ymin, xmax, ymax, value='', maxlen=1000000, multiline=0):
        doc = canvas._doc
        page = doc.thisPageRef()
        R, G, B = obj_R_G_B(canvas._fillColorObj)
        font = canvas._fontname
        fontsize = canvas._fontsize
        field = TextField(title, value, xmin, ymin, xmax, ymax, page, maxlen, font, fontsize, R, G, B, multiline)
        self.fields.append(field)
        canvas._addAnnotation(field)

    def selectField(self, canvas, title, value, options, xmin, ymin, xmax, ymax):
        doc = canvas._doc
        page = doc.thisPageRef()
        R, G, B = obj_R_G_B(canvas._fillColorObj)
        font = canvas._fontname
        fontsize = canvas._fontsize
        field = SelectField(title, value, options, xmin, ymin, xmax, ymax, page, font=font, fontsize=fontsize, R=R, G=G, B=B)
        self.fields.append(field)
        canvas._addAnnotation(field)

    def buttonField(self, canvas, title, value, xmin, ymin, width=16.7704, height=14.907):
        doc = canvas._doc
        page = doc.thisPageRef()
        field = ButtonField(title, value, xmin, ymin, page, width=width, height=height)
        self.fields.append(field)
        canvas._addAnnotation(field)

    def format(self, document):
        from reportlab.pdfbase.pdfdoc import PDFArray
        proxy = PDFPattern(FormPattern, Resources=getattr(self, 'resources', None) or FormResources(), NeedAppearances=getattr(self, 'needAppearances', 'false'), fields=PDFArray(self.fields), SigFlags=getattr(self, 'sigFlags', 0))
        return proxy.format(document)