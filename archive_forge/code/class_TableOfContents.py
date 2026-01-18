from reportlab.lib.units import cm
from reportlab.lib.utils import commasplit, escapeOnce, encode_label, decode_label, strTypes, asUnicode, asNative
from reportlab.lib.styles import ParagraphStyle, _baseFontName
from reportlab.lib import sequencer as rl_sequencer
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.doctemplate import IndexingFlowable
from reportlab.platypus.tables import TableStyle, Table
from reportlab.platypus.flowables import Spacer
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
import unicodedata
from ast import literal_eval
class TableOfContents(IndexingFlowable):
    """This creates a formatted table of contents.

    It presumes a correct block of data is passed in.
    The data block contains a list of (level, text, pageNumber)
    triplets.  You can supply a paragraph style for each level
    (starting at zero).
    Set dotsMinLevel to determine from which level on a line of
    dots should be drawn between the text and the page number.
    If dotsMinLevel is set to a negative value, no dotted lines are drawn.
    """

    def __init__(self, **kwds):
        self.rightColumnWidth = kwds.pop('rightColumnWidth', 72)
        self.levelStyles = kwds.pop('levelStyles', defaultLevelStyles)
        self.tableStyle = kwds.pop('tableStyle', defaultTableStyle)
        self.dotsMinLevel = kwds.pop('dotsMinLevel', 1)
        self.formatter = kwds.pop('formatter', None)
        if kwds:
            raise ValueError('unexpected keyword arguments %s' % ', '.join(kwds.keys()))
        self._table = None
        self._entries = []
        self._lastEntries = []

    def beforeBuild(self):
        self._lastEntries = self._entries[:]
        self.clearEntries()

    def isIndexing(self):
        return 1

    def isSatisfied(self):
        return self._entries == self._lastEntries

    def notify(self, kind, stuff):
        """The notification hook called to register all kinds of events.

        Here we are interested in 'TOCEntry' events only.
        """
        if kind == 'TOCEntry':
            self.addEntry(*stuff)

    def clearEntries(self):
        self._entries = []

    def getLevelStyle(self, n):
        """Returns the style for level n, generating and caching styles on demand if not present."""
        try:
            return self.levelStyles[n]
        except IndexError:
            prevstyle = self.getLevelStyle(n - 1)
            self.levelStyles.append(ParagraphStyle(name='%s-%d-indented' % (prevstyle.name, n), parent=prevstyle, firstLineIndent=prevstyle.firstLineIndent + delta, leftIndent=prevstyle.leftIndent + delta))
            return self.levelStyles[n]

    def addEntry(self, level, text, pageNum, key=None):
        """Adds one entry to the table of contents.

        This allows incremental buildup by a doctemplate.
        Requires that enough styles are defined."""
        assert type(level) == type(1), 'Level must be an integer'
        self._entries.append((level, text, pageNum, key))

    def addEntries(self, listOfEntries):
        """Bulk creation of entries in the table of contents.

        If you knew the titles but not the page numbers, you could
        supply them to get sensible output on the first run."""
        for entryargs in listOfEntries:
            self.addEntry(*entryargs)

    def wrap(self, availWidth, availHeight):
        """All table properties should be known by now."""
        if len(self._lastEntries) == 0:
            _tempEntries = [(0, 'Placeholder for table of contents', 0, None)]
        else:
            _tempEntries = self._lastEntries

        def drawTOCEntryEnd(canvas, kind, label):
            """Callback to draw dots and page numbers after each entry."""
            label = label.split(',')
            page, level, key = (int(label[0]), int(label[1]), literal_eval(label[2]))
            style = self.getLevelStyle(level)
            if self.dotsMinLevel >= 0 and level >= self.dotsMinLevel:
                dot = ' . '
            else:
                dot = ''
            if self.formatter:
                page = self.formatter(page)
            drawPageNumbers(canvas, style, [(page, key)], availWidth, availHeight, dot)
        self.canv.drawTOCEntryEnd = drawTOCEntryEnd
        tableData = []
        for level, text, pageNum, key in _tempEntries:
            style = self.getLevelStyle(level)
            if key:
                text = '<a href="#%s">%s</a>' % (key, text)
                keyVal = repr(key).replace(',', '\\x2c').replace('"', '\\x2c')
            else:
                keyVal = None
            para = Paragraph('%s<onDraw name="drawTOCEntryEnd" label="%d,%d,%s"/>' % (text, pageNum, level, keyVal), style)
            if style.spaceBefore:
                tableData.append([Spacer(1, style.spaceBefore)])
            tableData.append([para])
        self._table = Table(tableData, colWidths=(availWidth,), style=self.tableStyle)
        self.width, self.height = self._table.wrapOn(self.canv, availWidth, availHeight)
        return (self.width, self.height)

    def split(self, availWidth, availHeight):
        """At this stage we do not care about splitting the entries,
        we will just return a list of platypus tables.  Presumably the
        calling app has a pointer to the original TableOfContents object;
        Platypus just sees tables.
        """
        return self._table.splitOn(self.canv, availWidth, availHeight)

    def drawOn(self, canvas, x, y, _sW=0):
        """Don't do this at home!  The standard calls for implementing
        draw(); we are hooking this in order to delegate ALL the drawing
        work to the embedded table object.
        """
        self._table.drawOn(canvas, x, y, _sW)