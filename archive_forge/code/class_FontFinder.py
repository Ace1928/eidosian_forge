import sys, os, pickle
from hashlib import md5
from xml.sax.saxutils import quoteattr
from time import process_time as clock
from reportlab.lib.utils import asBytes, asNative as _asNative
from reportlab.lib.utils import rl_isdir, rl_isfile, rl_listdir, rl_getmtime
class FontFinder:

    def __init__(self, dirs=[], useCache=True, validate=False, recur=False, fsEncoding=None, verbose=0):
        self.useCache = useCache
        self.validate = validate
        if fsEncoding is None:
            fsEncoding = sys.getfilesystemencoding()
        self._fsEncoding = fsEncoding or 'utf8'
        self._dirs = set()
        self._recur = recur
        self.addDirectories(dirs)
        self._fonts = []
        self._skippedFiles = []
        self._badFiles = []
        self._fontsByName = {}
        self._fontsByFamily = {}
        self._fontsByFamilyBoldItalic = {}
        self.verbose = verbose

    def addDirectory(self, dirName, recur=None):
        if rl_isdir(dirName):
            self._dirs.add(dirName)
            if recur if recur is not None else self._recur:
                for r, D, F in os.walk(dirName):
                    for d in D:
                        self._dirs.add(os.path.join(r, d))

    def addDirectories(self, dirNames, recur=None):
        for dirName in dirNames:
            self.addDirectory(dirName, recur=recur)

    def getFamilyNames(self):
        """Returns a list of the distinct font families found"""
        if not self._fontsByFamily:
            fonts = self._fonts
            for font in fonts:
                fam = font.familyName
                if fam is None:
                    continue
                if fam in self._fontsByFamily:
                    self._fontsByFamily[fam].append(font)
                else:
                    self._fontsByFamily[fam] = [font]
        fsEncoding = self._fsEncoding
        names = list((asBytes(_, enc=fsEncoding) for _ in self._fontsByFamily.keys()))
        names.sort()
        return names

    def getFontsInFamily(self, familyName):
        """Return list of all font objects with this family name"""
        return self._fontsByFamily.get(familyName, [])

    def getFamilyXmlReport(self):
        """Reports on all families found as XML.
        """
        lines = []
        lines.append('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
        lines.append('<font_families>')
        for dirName in self._dirs:
            lines.append('    <directory name=%s/>' % quoteattr(asNative(dirName)))
        for familyName in self.getFamilyNames():
            if familyName:
                lines.append('    <family name=%s>' % quoteattr(asNative(familyName)))
                for font in self.getFontsInFamily(familyName):
                    lines.append('        ' + font.getTag())
                lines.append('    </family>')
        lines.append('</font_families>')
        return '\n'.join(lines)

    def getFontsWithAttributes(self, **kwds):
        """This is a general lightweight search."""
        selected = []
        for font in self._fonts:
            OK = True
            for k, v in kwds.items():
                if getattr(font, k, None) != v:
                    OK = False
            if OK:
                selected.append(font)
        return selected

    def getFont(self, familyName, bold=False, italic=False):
        """Try to find a font matching the spec"""
        for font in self._fonts:
            if font.familyName == familyName:
                if font.isBold == bold:
                    if font.isItalic == italic:
                        return font
        raise KeyError('Cannot find font %s with bold=%s, italic=%s' % (familyName, bold, italic))

    def _getCacheFileName(self):
        """Base this on the directories...same set of directories
        should give same cache"""
        fsEncoding = self._fsEncoding
        hash = md5(b''.join((asBytes(_, enc=fsEncoding) for _ in sorted(self._dirs)))).hexdigest()
        from reportlab.lib.utils import get_rl_tempfile
        fn = get_rl_tempfile('fonts_%s.dat' % hash)
        return fn

    def save(self, fileName):
        f = open(fileName, 'wb')
        pickle.dump(self, f)
        f.close()

    def load(self, fileName):
        f = open(fileName, 'rb')
        finder2 = pickle.load(f)
        f.close()
        self.__dict__.update(finder2.__dict__)

    def search(self):
        if self.verbose:
            started = clock()
        if not self._dirs:
            raise ValueError('Font search path is empty!  Please specify search directories using addDirectory or addDirectories')
        if self.useCache:
            cfn = self._getCacheFileName()
            if rl_isfile(cfn):
                try:
                    self.load(cfn)
                    if self.verbose >= 3:
                        print('loaded cached file with %d fonts (%s)' % (len(self._fonts), cfn))
                    return
                except:
                    pass
        for dirName in self._dirs:
            try:
                fileNames = rl_listdir(dirName)
            except:
                continue
            for fileName in fileNames:
                root, ext = os.path.splitext(fileName)
                if ext.lower() in EXTENSIONS:
                    f = FontDescriptor()
                    f.fileName = fileName = os.path.normpath(os.path.join(dirName, fileName))
                    try:
                        f.timeModified = rl_getmtime(fileName)
                    except:
                        self._skippedFiles.append(fileName)
                        continue
                    ext = ext.lower()
                    if ext[0] == '.':
                        ext = ext[1:]
                    f.typeCode = ext
                    if ext in ('otf', 'pfa'):
                        self._skippedFiles.append(fileName)
                    elif ext in ('ttf', 'ttc'):
                        from reportlab.pdfbase.ttfonts import TTFontFile, TTFError
                        try:
                            font = TTFontFile(fileName, validate=self.validate)
                        except TTFError:
                            self._badFiles.append(fileName)
                            continue
                        f.name = font.name
                        f.fullName = font.fullName
                        f.styleName = font.styleName
                        f.familyName = font.familyName
                        f.isBold = FF_FORCEBOLD == FF_FORCEBOLD & font.flags
                        f.isItalic = FF_ITALIC == FF_ITALIC & font.flags
                    elif ext == 'pfb':
                        if rl_isfile(os.path.join(dirName, root + '.afm')):
                            f.metricsFileName = os.path.normpath(os.path.join(dirName, root + '.afm'))
                        elif rl_isfile(os.path.join(dirName, root + '.AFM')):
                            f.metricsFileName = os.path.normpath(os.path.join(dirName, root + '.AFM'))
                        else:
                            self._skippedFiles.append(fileName)
                            continue
                        from reportlab.pdfbase.pdfmetrics import parseAFMFile
                        info, glyphs = parseAFMFile(f.metricsFileName)
                        f.name = info['FontName']
                        f.fullName = info.get('FullName', f.name)
                        f.familyName = info.get('FamilyName', None)
                        f.isItalic = float(info.get('ItalicAngle', 0)) > 0.0
                        f.isBold = 'bold' in info.get('Weight', '').lower()
                    self._fonts.append(f)
        if self.useCache:
            self.save(cfn)
        if self.verbose:
            finished = clock()
            print('found %d fonts; skipped %d; bad %d.  Took %0.2f seconds' % (len(self._fonts), len(self._skippedFiles), len(self._badFiles), finished - started))