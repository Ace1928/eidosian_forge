import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
def addPostScriptCommand(self, command, position=1):
    """Embed literal Postscript in the document.

        With position=0, it goes at very beginning of page stream;
        with position=1, at current point; and
        with position=2, at very end of page stream.  What that does
        to the resulting Postscript depends on Adobe's header :-)

        Use with extreme caution, but sometimes needed for printer tray commands.
        Acrobat 4.0 will export Postscript to a printer or file containing
        the given commands.  Adobe Reader 6.0 no longer does as this feature is
        deprecated.  5.0, I don't know about (please let us know!). This was
        funded by Bob Marshall of Vector.co.uk and tested on a Lexmark 750.
        See test_pdfbase_postscript.py for 2 test cases - one will work on
        any Postscript device, the other uses a 'setpapertray' command which
        will error in Distiller but work on printers supporting it.
        """
    if isUnicode(command):
        rawName = 'PS' + hashlib.md5(command.encode('utf-8')).hexdigest()
    else:
        rawName = 'PS' + hashlib.md5(command).hexdigest()
    regName = self._doc.getXObjectName(rawName)
    psObj = self._doc.idToObject.get(regName, None)
    if not psObj:
        psObj = pdfdoc.PDFPostScriptXObject(command + '\n')
        self._setXObjects(psObj)
        self._doc.Reference(psObj, regName)
        self._doc.addForm(rawName, psObj)
    if position == 0:
        self._psCommandsBeforePage.append('/%s Do' % regName)
    elif position == 1:
        self._code.append('/%s Do' % regName)
    else:
        self._psCommandsAfterPage.append('/%s Do' % regName)
    self._formsinuse.append(rawName)