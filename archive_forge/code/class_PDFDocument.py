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
class PDFDocument(PDFObject):
    defaultStreamFilters = None
    encrypt = NoEncryption()

    def __init__(self, dummyoutline=0, compression=rl_config.pageCompression, invariant=rl_config.invariant, filename=None, pdfVersion=PDF_VERSION_DEFAULT, lang=None):
        self._ID = None
        self.objectcounter = 0
        self.shadingCounter = 0
        self.inObject = None
        self.pageCounter = 1
        if invariant is None:
            self.invariant = rl_config.invariant
        else:
            self.invariant = invariant
        self.setCompression(compression)
        self._pdfVersion = pdfVersion
        sig = self.signature = md5()
        sig.update(b'a reportlab document')
        self._timeStamp = TimeStamp(self.invariant)
        cat = self._timeStamp.t
        cat = ascii(cat)
        sig.update(bytestr(cat))
        self.idToObjectNumberAndVersion = {}
        self.idToObject = {}
        self.idToOffset = {}
        self.numberToId = {}
        cat = self.Catalog = self._catalog = PDFCatalog()
        pages = self.Pages = PDFPages()
        cat.Pages = pages
        lang = lang if lang else rl_config.documentLang
        if lang:
            cat.Lang = PDFString(lang)
        self.outline = self.Outlines = cat.Outlines = PDFOutlines0() if dummyoutline else PDFOutlines()
        self.info = PDFInfo()
        self.fontMapping = {}
        DD = PDFDictionary({})
        DD.__Comment__ = 'The standard fonts dictionary'
        self.Reference(DD, BasicFonts)
        self.delayedFonts = []

    def setCompression(self, onoff):
        self.compression = onoff

    def ensureMinPdfVersion(self, *keys):
        """Ensure that the pdf version is greater than or equal to that specified by the keys"""
        for k in keys:
            self._pdfVersion = max(self._pdfVersion, PDF_SUPPORT_VERSION[k])

    def updateSignature(self, thing):
        """add information to the signature"""
        if self._ID:
            return
        self.signature.update(bytestr(thing))

    def ID(self):
        """A unique fingerprint for the file (unless in invariant mode)"""
        if self._ID:
            return self._ID
        digest = self.signature.digest()
        doc = DummyDoc()
        IDs = PDFText(digest, enc='raw').format(doc)
        self._ID = b'\n[' + IDs + IDs + b']\n% ReportLab generated PDF document -- digest (http://www.reportlab.com)\n'
        return self._ID

    def SaveToFile(self, filename, canvas):
        if getattr(self, '_savedToFile', False):
            raise RuntimeError('class %s instances can only be saved once' % self.__class__.__name__)
        self._savedToFile = True
        if hasattr(getattr(filename, 'write', None), '__call__'):
            myfile = 0
            f = filename
            filename = getattr(f, 'name', None)
            if isinstance(filename, int):
                filename = '<os fd:%d>' % filename
            elif not isStr(filename):
                filename = '<%s@0X%8.8X>' % (f.__class__.__name__, id(f))
            filename = makeFileName(filename)
        elif isStr(filename):
            myfile = 1
            filename = makeFileName(filename)
            f = open(filename, 'wb')
        else:
            raise TypeError('Cannot use %s as a filename or file' % repr(filename))
        data = self.GetPDFData(canvas)
        if isUnicode(data):
            data = data.encode('latin1')
        f.write(data)
        if myfile:
            f.close()
            import os
            if os.name == 'mac':
                from reportlab.lib.utils import markfilename
                markfilename(filename)
        if getattr(canvas, '_verbosity', None):
            print('saved %s' % (filename,))

    def GetPDFData(self, canvas):
        for fnt in self.delayedFonts:
            fnt.addObjects(self)
        self.info.invariant = self.invariant
        self.info.digest(self.signature)
        self.Reference(self.Catalog)
        self.Reference(self.info)
        self.Outlines.prepare(self, canvas)
        if self.Outlines.ready < 0:
            self.Catalog.Outlines = None
        return self.format()

    def inPage(self):
        """specify the current object as a page (enables reference binding and other page features)"""
        if self.inObject is not None:
            if self.inObject == 'page':
                return
            raise ValueError("can't go in page already in object %s" % self.inObject)
        self.inObject = 'page'

    def inForm(self):
        """specify that we are in a form xobject (disable page features, etc)"""
        self.inObject = 'form'

    def getInternalFontName(self, psfontname):
        fm = self.fontMapping
        if psfontname in fm:
            return fm[psfontname]
        else:
            try:
                fontObj = pdfmetrics.getFont(psfontname)
                if fontObj._dynamicFont:
                    raise PDFError('getInternalFontName(%s) called for a dynamic font' % repr(psfontname))
                fontObj.addObjects(self)
                return fm[psfontname]
            except KeyError:
                raise PDFError('Font %s not known!' % repr(psfontname))

    def thisPageName(self):
        return 'Page' + repr(self.pageCounter)

    def thisPageRef(self):
        return PDFObjectReference(self.thisPageName())

    def addPage(self, page):
        name = self.thisPageName()
        self.Reference(page, name)
        self.Pages.addPage(page)
        self.pageCounter += 1
        self.inObject = None

    def addForm(self, name, form):
        """add a Form XObject."""
        if self.inObject != 'form':
            self.inForm()
        self.Reference(form, xObjectName(name))
        self.inObject = None

    def annotationName(self, externalname):
        return 'Annot.%s' % externalname

    def addAnnotation(self, name, annotation):
        self.Reference(annotation, self.annotationName(name))

    def refAnnotation(self, name):
        internalname = self.annotationName(name)
        return PDFObjectReference(internalname)

    def addShading(self, shading):
        name = 'Sh%d' % self.shadingCounter
        self.Reference(shading, name)
        self.shadingCounter += 1
        return name

    def addColor(self, cmyk):
        sname = cmyk.spotName
        if not sname:
            if cmyk.cyan == 0 and cmyk.magenta == 0 and (cmyk.yellow == 0):
                sname = 'BLACK'
            elif cmyk.black == 0 and cmyk.magenta == 0 and (cmyk.yellow == 0):
                sname = 'CYAN'
            elif cmyk.cyan == 0 and cmyk.black == 0 and (cmyk.yellow == 0):
                sname = 'MAGENTA'
            elif cmyk.cyan == 0 and cmyk.magenta == 0 and (cmyk.black == 0):
                sname = 'YELLOW'
            if not sname:
                raise ValueError('CMYK colour %r used without a spotName' % cmyk)
            else:
                cmyk = cmyk.clone(spotName=sname)
        name = PDFName(sname)[1:]
        if name not in self.idToObject:
            sep = PDFSeparationCMYKColor(cmyk).value()
            self.Reference(sep, name)
        return (name, sname)

    def setTitle(self, title):
        """embeds in PDF file"""
        if title is None:
            self.info.title = '(anonymous)'
        else:
            self.info.title = title

    def setAuthor(self, author):
        """embedded in PDF file"""
        if author is None:
            self.info.author = '(anonymous)'
        else:
            self.info.author = author

    def setSubject(self, subject):
        """embeds in PDF file"""
        if subject is None:
            self.info.subject = '(unspecified)'
        else:
            self.info.subject = subject

    def setCreator(self, creator):
        """embeds in PDF file"""
        if creator is None:
            self.info.creator = '(unspecified)'
        else:
            self.info.creator = creator

    def setProducer(self, producer):
        """embeds in PDF file"""
        if producer is None:
            self.info.producer = _default_producer
        else:
            self.info.producer = producer

    def setKeywords(self, keywords):
        """embeds a string containing keywords in PDF file"""
        if keywords is None:
            self.info.keywords = ''
        else:
            self.info.keywords = keywords

    def setDateFormatter(self, dateFormatter):
        self.info._dateFormatter = dateFormatter

    def getAvailableFonts(self):
        fontnames = list(self.fontMapping.keys())
        from reportlab.pdfbase import _fontdata
        for name in _fontdata.standardFonts:
            if name not in fontnames:
                fontnames.append(name)
        fontnames.sort()
        return fontnames

    def format(self):
        self.encrypt.prepare(self)
        cat = self.Catalog
        info = self.info
        self.Reference(cat)
        self.Reference(info)
        encryptref = None
        encryptinfo = self.encrypt.info()
        if encryptinfo:
            encryptref = self.Reference(encryptinfo)
        counter = 0
        ids = []
        numbertoid = self.numberToId
        idToNV = self.idToObjectNumberAndVersion
        idToOb = self.idToObject
        idToOf = self.idToOffset
        self.__accum__ = File = PDFFile(self._pdfVersion)
        while True:
            counter += 1
            if counter not in numbertoid:
                break
            oid = numbertoid[counter]
            obj = idToOb[oid]
            IO = PDFIndirectObject(oid, obj)
            IOf = IO.format(self)
            if not rl_config.invariant and rl_config.pdfComments:
                try:
                    classname = obj.__class__.__name__
                except:
                    classname = ascii(obj)
                File.add('%% %s: class %s \n' % (ascii(oid), classname[:50]))
            offset = File.add(IOf)
            idToOf[oid] = offset
            ids.append(oid)
        del self.__accum__
        lno = len(numbertoid)
        if counter - 1 != lno:
            raise ValueError("counter %s doesn't match number to id dictionary %s" % (counter, lno))
        xref = PDFCrossReferenceTable()
        xref.addsection(0, ids)
        xreff = xref.format(self)
        xrefoffset = File.add(xreff)
        trailer = PDFTrailer(startxref=xrefoffset, Size=lno + 1, Root=self.Reference(cat), Info=self.Reference(info), Encrypt=encryptref, ID=self.ID())
        trailerf = trailer.format(self)
        File.add(trailerf)
        for ds in getattr(self, '_digiSigs', []):
            ds.sign(File)
        return File.format(self)

    def hasForm(self, name):
        """test for existence of named form"""
        internalname = xObjectName(name)
        return internalname in self.idToObject

    def getFormBBox(self, name, boxType='MediaBox'):
        """get the declared bounding box of the form as a list.
        If you specify a different PDF box definition (e.g. the
        ArtBox) and it has one, that's what you'll get."""
        internalname = xObjectName(name)
        if internalname in self.idToObject:
            theform = self.idToObject[internalname]
            if hasattr(theform, '_extra_pageCatcher_info'):
                return theform._extra_pageCatcher_info[boxType]
            if isinstance(theform, PDFFormXObject):
                return theform.BBoxList()
            elif isinstance(theform, PDFStream):
                return list(theform.dictionary.dict[boxType].sequence)
            else:
                raise ValueError("I don't understand the form instance %s" % repr(name))

    def getXObjectName(self, name):
        """Lets canvas find out what form is called internally.
        Never mind whether it is defined yet or not."""
        return xObjectName(name)

    def xobjDict(self, formnames):
        """construct an xobject dict (for inclusion in a resource dict, usually)
           from a list of form names (images not yet supported)"""
        D = {}
        for name in formnames:
            internalname = xObjectName(name)
            reference = PDFObjectReference(internalname)
            D[internalname] = reference
        return PDFDictionary(D)

    def Reference(self, obj, name=None):
        iob = isinstance(obj, PDFObject)
        idToObject = self.idToObject
        if name is None and (not iob or obj.__class__ is PDFObjectReference):
            return obj
        if hasattr(obj, __InternalName__):
            intname = obj.__InternalName__
            if name is not None and name != intname:
                raise ValueError('attempt to reregister object %s with new name %s' % (repr(intname), repr(name)))
            if intname not in idToObject:
                raise ValueError('object of type %s named as %s, but not registered' % (type(obj), ascii(intname)))
            return PDFObjectReference(intname)
        objectcounter = self.objectcounter = self.objectcounter + 1
        if name is None:
            name = 'R' + repr(objectcounter)
        if name in idToObject:
            other = idToObject[name]
            if other != obj:
                raise ValueError('redefining named object: ' + repr(name))
            return PDFObjectReference(name)
        if iob:
            obj.__InternalName__ = name
        self.idToObjectNumberAndVersion[name] = (objectcounter, 0)
        self.numberToId[objectcounter] = name
        idToObject[name] = obj
        return PDFObjectReference(name)