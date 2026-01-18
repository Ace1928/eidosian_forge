import binascii, codecs, zlib
from collections import OrderedDict
from reportlab.pdfbase import pdfutils
from reportlab import rl_config
from reportlab.lib.utils import open_for_read, makeFileName, isSeq, isBytes, isUnicode, _digester, isStr, bytestr, annotateException, TimeStamp
from reportlab.lib.rl_accel import escapePDF, fp_str, asciiBase85Encode, asciiBase85Decode
from reportlab.pdfbase import pdfmetrics
from hashlib import md5
from sys import stderr
import re
class PDFOutlines(PDFObject):
    """
    takes a recursive list of outline destinations like::

        out = PDFOutline1()
        out.setNames(canvas, # requires canvas for name resolution
        "chapter1dest",
        ("chapter2dest",
        ["chapter2section1dest",
        "chapter2section2dest",
        "chapter2conclusiondest"]
        ), # end of chapter2 description
        "chapter3dest",
        ("chapter4dest", ["c4s1", "c4s2"])
        )

    Higher layers may build this structure incrementally. KISS at base level.
    """
    mydestinations = ready = None
    counter = 0
    currentlevel = -1

    def __init__(self):
        self.destinationnamestotitles = {}
        self.destinationstotitles = {}
        self.levelstack = []
        self.buildtree = []
        self.closedict = {}

    def addOutlineEntry(self, destinationname, level=0, title=None, closed=None):
        """destinationname of None means "close the tree" """
        if destinationname is None and level != 0:
            raise ValueError('close tree must have level of 0')
        if not isinstance(level, int):
            raise ValueError('level must be integer, got %s' % type(level))
        if level < 0:
            raise ValueError('negative levels not allowed')
        if title is None:
            title = destinationname
        currentlevel = self.currentlevel
        stack = self.levelstack
        tree = self.buildtree
        if level > currentlevel:
            if level > currentlevel + 1:
                raise ValueError("can't jump from outline level %s to level %s, need intermediates (destinationname=%r, title=%r)" % (currentlevel, level, destinationname, title))
            level = currentlevel = currentlevel + 1
            stack.append([])
        while level < currentlevel:
            current = stack[-1]
            del stack[-1]
            previous = stack[-1]
            lastinprevious = previous[-1]
            if isinstance(lastinprevious, tuple):
                name, sectionlist = lastinprevious
                raise ValueError('cannot reset existing sections: ' + repr(lastinprevious))
            else:
                name = lastinprevious
                sectionlist = current
                previous[-1] = (name, sectionlist)
            currentlevel = currentlevel - 1
        if destinationname is None:
            return
        stack[-1].append(destinationname)
        self.destinationnamestotitles[destinationname] = title
        if closed:
            self.closedict[destinationname] = 1
        self.currentlevel = level

    def setDestinations(self, destinationtree):
        self.mydestinations = destinationtree

    def format(self, document):
        D = {}
        D['Type'] = PDFName('Outlines')
        D['Count'] = self.count
        D['First'] = self.first
        D['Last'] = self.last
        PD = PDFDictionary(D)
        return PD.format(document)

    def setNames(self, canvas, *nametree):
        desttree = self.translateNames(canvas, nametree)
        self.setDestinations(desttree)

    def setNameList(self, canvas, nametree):
        """Explicit list so I don't need to do in the caller"""
        desttree = self.translateNames(canvas, nametree)
        self.setDestinations(desttree)

    def translateNames(self, canvas, object):
        """recursively translate tree of names into tree of destinations"""
        destinationnamestotitles = self.destinationnamestotitles
        destinationstotitles = self.destinationstotitles
        closedict = self.closedict
        if isStr(object):
            if not isUnicode(object):
                object = object.decode('utf8')
            destination = canvas._bookmarkReference(object)
            title = object
            if object in destinationnamestotitles:
                title = destinationnamestotitles[object]
            else:
                destinationnamestotitles[title] = title
            destinationstotitles[destination] = title
            if object in closedict:
                closedict[destination] = 1
            return {object: canvas._bookmarkReference(object)}
        if isSeq(object):
            L = []
            for o in object:
                L.append(self.translateNames(canvas, o))
            if isinstance(object, tuple):
                return tuple(L)
            return L
        raise TypeError('in outline, destination name must be string: got a %s' % type(object))

    def prepare(self, document, canvas):
        """prepare all data structures required for save operation (create related objects)"""
        if self.mydestinations is None:
            if self.levelstack:
                self.addOutlineEntry(None)
                destnames = self.levelstack[0]
                self.mydestinations = self.translateNames(canvas, destnames)
            else:
                self.first = self.last = None
                self.count = 0
                self.ready = -1
                return
        self.count = count(self.mydestinations, self.closedict)
        self.first, self.last = self.maketree(document, self.mydestinations, toplevel=1)
        self.ready = 1

    def maketree(self, document, destinationtree, Parent=None, toplevel=0):
        if toplevel:
            levelname = 'Outline'
            Parent = document.Reference(document.Outlines)
        else:
            self.count += 1
            levelname = 'Outline.%s' % self.count
            if Parent is None:
                raise ValueError('non-top level outline elt parent must be specified')
        if not isSeq(destinationtree):
            raise ValueError('destinationtree must be list or tuple, got %s')
        nelts = len(destinationtree)
        lastindex = nelts - 1
        lastelt = firstref = lastref = None
        destinationnamestotitles = self.destinationnamestotitles
        closedict = self.closedict
        for index in range(nelts):
            eltobj = OutlineEntryObject()
            eltobj.Parent = Parent
            eltname = '%s.%s' % (levelname, index)
            eltref = document.Reference(eltobj, eltname)
            if lastelt is not None:
                lastelt.Next = eltref
                eltobj.Prev = lastref
            if firstref is None:
                firstref = eltref
            lastref = eltref
            lastelt = eltobj
            lastref = eltref
            elt = destinationtree[index]
            if isinstance(elt, dict):
                leafdict = elt
            elif isinstance(elt, tuple):
                try:
                    leafdict, subsections = elt
                except:
                    raise ValueError('destination tree elt tuple should have two elts, got %s' % len(elt))
                eltobj.Count = count(subsections, closedict)
                eltobj.First, eltobj.Last = self.maketree(document, subsections, eltref)
            else:
                raise ValueError('destination tree elt should be dict or tuple, got %s' % type(elt))
            try:
                [(Title, Dest)] = list(leafdict.items())
            except:
                raise ValueError('bad outline leaf dictionary, should have one entry ' + bytestr(elt))
            eltobj.Title = destinationnamestotitles[Title]
            eltobj.Dest = Dest
            if isinstance(elt, tuple) and Dest in closedict:
                eltobj.Count = -eltobj.Count
        return (firstref, lastref)