from reportlab.platypus.flowables import *
from reportlab.platypus.flowables import _ContainerSpace
from reportlab.lib.units import inch
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.frames import Frame
from reportlab.rl_config import defaultPageSize, verbose
import reportlab.lib.sequencer
from reportlab.pdfgen import canvas
from reportlab.lib.utils import isSeq, encode_label, decode_label, annotateException, strTypes
import sys
import logging
class BaseDocTemplate:
    """
    First attempt at defining a document template class.

    The basic idea is simple.

    1)  The document has a list of data associated with it
        this data should derive from flowables. We'll have
        special classes like PageBreak, FrameBreak to do things
        like forcing a page end etc.

    2)  The document has one or more page templates.

    3)  Each page template has one or more frames.

    4)  The document class provides base methods for handling the
        story events and some reasonable methods for getting the
        story flowables into the frames.

    5)  The document instances can override the base handler routines.

    Most of the methods for this class are not called directly by the user,
    but in some advanced usages they may need to be overridden via subclassing.

    EXCEPTION: doctemplate.build(...) must be called for most reasonable uses
    since it builds a document using the page template.

    Each document template builds exactly one document into a file specified
    by the filename argument on initialization.

    Possible keyword arguments for the initialization:

    - pageTemplates: A list of templates.  Must be nonempty.  Names
      assigned to the templates are used for referring to them so no two used
      templates should have the same name.  For example you might want one template
      for a title page, one for a section first page, one for a first page of
      a chapter and two more for the interior of a chapter on odd and even pages.
      If this argument is omitted then at least one pageTemplate should be provided
      using the addPageTemplates method before the document is built.
    - pageSize: a 2-tuple or a size constant from reportlab/lib/pagesizes.pu.
      Used by the SimpleDocTemplate subclass which does NOT accept a list of
      pageTemplates but makes one for you; ignored when using pageTemplates.

    - showBoundary: if set draw a box around the frame boundaries.
    - leftMargin:
    - rightMargin:
    - topMargin:
    - bottomMargin:  Margin sizes in points (default 1 inch).  These margins may be
      overridden by the pageTemplates.  They are primarily of interest for the
      SimpleDocumentTemplate subclass.

    - allowSplitting:  If set flowables (eg, paragraphs) may be split across frames or pages
      (default: 1)
    - title: Internal title for document (does not automatically display on any page)
    - author: Internal author for document (does not automatically display on any page)
    """
    _initArgs = {'pagesize': defaultPageSize, 'pageTemplates': [], 'showBoundary': 0, 'leftMargin': inch, 'rightMargin': inch, 'topMargin': inch, 'bottomMargin': inch, 'allowSplitting': 1, 'title': None, 'author': None, 'subject': None, 'creator': None, 'producer': None, 'keywords': [], 'invariant': None, 'pageCompression': None, '_pageBreakQuick': 1, 'rotation': 0, '_debug': 0, 'encrypt': None, 'cropMarks': None, 'enforceColorSpace': None, 'displayDocTitle': None, 'lang': None, 'initialFontName': None, 'initialFontSize': None, 'initialLeading': None, 'cropBox': None, 'artBox': None, 'trimBox': None, 'bleedBox': None, 'keepTogetherClass': KeepTogether, 'hideToolbar': None, 'hideMenubar': None, 'hideWindowUI': None, 'fitWindow': None, 'centerWindow': None, 'nonFullScreenPageMode': None, 'direction': None, 'viewArea': None, 'viewClip': None, 'printArea': None, 'printClip': None, 'printScaling': None, 'duplex': None}
    _invalidInitArgs = ()
    _firstPageTemplateIndex = 0

    def __init__(self, filename, **kw):
        """create a document template bound to a filename (see class documentation for keyword arguments)"""
        self.filename = filename
        self._nameSpace = dict(doc=self)
        self._lifetimes = {}
        for k in self._initArgs.keys():
            if k not in kw:
                v = self._initArgs[k]
            else:
                if k in self._invalidInitArgs:
                    raise ValueError('Invalid argument %s' % k)
                v = kw[k]
            setattr(self, k, v)
        p = self.pageTemplates
        self.pageTemplates = []
        self.addPageTemplates(p)
        self._pageRefs = {}
        self._indexingFlowables = []
        self._onPage = None
        self._onProgress = None
        self._flowableCount = 0
        self._curPageFlowableCount = 0
        self._emptyPages = 0
        self._emptyPagesAllowed = 10
        self._leftExtraIndent = 0.0
        self._rightExtraIndent = 0.0
        self._topFlowables = []
        self._pageTopFlowables = []
        self._frameBGs = []
        self._calc()
        self.afterInit()

    def _calc(self):
        self._rightMargin = self.pagesize[0] - self.rightMargin
        self._topMargin = self.pagesize[1] - self.topMargin
        self.width = self._rightMargin - self.leftMargin
        self.height = self._topMargin - self.bottomMargin

    def setPageCallBack(self, func):
        """Simple progress monitor - func(pageNo) called on each new page"""
        self._onPage = func

    def setProgressCallBack(self, func):
        """Cleverer progress monitor - func(typ, value) called regularly"""
        self._onProgress = func

    def clean_hanging(self):
        """handle internal postponed actions"""
        while len(self._hanging):
            self.handle_flowable(self._hanging)

    def addPageTemplates(self, pageTemplates):
        """add one or a sequence of pageTemplates"""
        if not isSeq(pageTemplates):
            pageTemplates = [pageTemplates]
        for t in pageTemplates:
            self.pageTemplates.append(t)

    def handle_documentBegin(self):
        """implement actions at beginning of document"""
        self._hanging = [PageBegin]
        if isinstance(self._firstPageTemplateIndex, list):
            self.handle_nextPageTemplate(self._firstPageTemplateIndex)
            self._setPageTemplate()
        else:
            self.pageTemplate = self.pageTemplates[self._firstPageTemplateIndex]
        self.page = 0
        self.beforeDocument()

    def handle_pageBegin(self):
        """Perform actions required at beginning of page.
        shouldn't normally be called directly"""
        self.page += 1
        if self._debug:
            logger.debug('beginning page %d' % self.page)
        self.pageTemplate.beforeDrawPage(self.canv, self)
        self.pageTemplate.checkPageSize(self.canv, self)
        self.pageTemplate.onPage(self.canv, self)
        for f in self.pageTemplate.frames:
            f._reset()
        self.beforePage()
        self._curPageFlowableCount = 0
        if hasattr(self, '_nextFrameIndex'):
            del self._nextFrameIndex
        self.frame = self.pageTemplate.frames[0]
        self.frame._debug = self._debug
        self.handle_frameBegin(pageTopFlowables=self._pageTopFlowables)

    def _setPageTemplate(self):
        if hasattr(self, '_nextPageTemplateCycle'):
            self.pageTemplate = self._nextPageTemplateCycle.next_value
        elif hasattr(self, '_nextPageTemplateIndex'):
            self.pageTemplate = self.pageTemplates[self._nextPageTemplateIndex]
            del self._nextPageTemplateIndex
        elif self.pageTemplate.autoNextPageTemplate:
            self.handle_nextPageTemplate(self.pageTemplate.autoNextPageTemplate)
            self.pageTemplate = self.pageTemplates[self._nextPageTemplateIndex]

    def _samePT(self, npt):
        if isSeq(npt):
            return getattr(self, '_nextPageTemplateCycle', [])
        if isinstance(npt, strTypes):
            return npt == (self.pageTemplates[self._nextPageTemplateIndex].id if hasattr(self, '_nextPageTemplateIndex') else self.pageTemplate.id)
        if isinstance(npt, int) and 0 <= npt < len(self.pageTemplates):
            if hasattr(self, '_nextPageTemplateIndex'):
                return npt == self._nextPageTemplateIndex
            return npt == self.pageTemplates.find(self.pageTemplate)

    def handle_pageEnd(self):
        """ show the current page
            check the next page template
            hang a page begin
        """
        self._removeVars(('page', 'frame'))
        self._leftExtraIndent = self.frame._leftExtraIndent
        self._rightExtraIndent = self.frame._rightExtraIndent
        self._frameBGs = self.frame._frameBGs
        if self._curPageFlowableCount == 0:
            self._emptyPages += 1
        else:
            self._emptyPages = 0
        if self._emptyPages >= self._emptyPagesAllowed:
            if 1:
                ident = 'More than %d pages generated without content - halting layout.  Likely that a flowable is too large for any frame.' % self._emptyPagesAllowed
                raise LayoutError(ident)
            else:
                pass
        else:
            if self._onProgress:
                self._onProgress('PAGE', self.canv.getPageNumber())
            self.pageTemplate.afterDrawPage(self.canv, self)
            self.pageTemplate.onPageEnd(self.canv, self)
            self.afterPage()
            if self._debug:
                logger.debug('ending page %d' % self.page)
            self.canv.setPageRotation(getattr(self.pageTemplate, 'rotation', self.rotation))
            self.canv.showPage()
            self._setPageTemplate()
            if self._emptyPages == 0:
                pass
        self._hanging.append(PageBegin)

    def handle_pageBreak(self, slow=None):
        """some might choose not to end all the frames"""
        if self._pageBreakQuick and (not slow):
            self.handle_pageEnd()
        else:
            n = len(self._hanging)
            while len(self._hanging) == n:
                self.handle_frameEnd()

    def handle_frameBegin(self, resume=0, pageTopFlowables=None):
        """What to do at the beginning of a frame"""
        f = self.frame
        if f._atTop:
            boundary = self.frame.showBoundary or self.showBoundary
            if boundary:
                self.frame.drawBoundary(self.canv, boundary)
        f._leftExtraIndent = self._leftExtraIndent
        f._rightExtraIndent = self._rightExtraIndent
        f._frameBGs = self._frameBGs
        if pageTopFlowables:
            self._hanging.extend(pageTopFlowables)
        if self._topFlowables:
            self._hanging.extend(self._topFlowables)

    def handle_frameEnd(self, resume=0):
        """ Handles the semantics of the end of a frame. This includes the selection of
            the next frame or if this is the last frame then invoke pageEnd.
        """
        self._removeVars(('frame',))
        self._leftExtraIndent = self.frame._leftExtraIndent
        self._rightExtraIndent = self.frame._rightExtraIndent
        self._frameBGs = self.frame._frameBGs
        if hasattr(self, '_nextFrameIndex'):
            self.frame = self.pageTemplate.frames[self._nextFrameIndex]
            self.frame._debug = self._debug
            del self._nextFrameIndex
            self.handle_frameBegin(resume)
        else:
            f = self.frame
            if hasattr(f, 'lastFrame') or f is self.pageTemplate.frames[-1]:
                self.handle_pageEnd()
                self.frame = None
            else:
                self.frame = self.pageTemplate.frames[self.pageTemplate.frames.index(f) + 1]
                self.frame._debug = self._debug
                self.handle_frameBegin()

    def handle_nextPageTemplate(self, pt):
        """On endPage change to the page template with name or index pt"""
        if isinstance(pt, strTypes):
            if hasattr(self, '_nextPageTemplateCycle'):
                del self._nextPageTemplateCycle
            for t in self.pageTemplates:
                if t.id == pt:
                    self._nextPageTemplateIndex = self.pageTemplates.index(t)
                    return
            raise ValueError("can't find template('%s')" % pt)
        elif isinstance(pt, int):
            if hasattr(self, '_nextPageTemplateCycle'):
                del self._nextPageTemplateCycle
            self._nextPageTemplateIndex = pt
        elif isSeq(pt):
            c = PTCycle()
            for ptn in pt:
                found = 0
                if ptn == '*':
                    c._restart = len(c)
                    continue
                for t in self.pageTemplates:
                    if t.id == ptn:
                        c.append(t)
                        found = 1
                if not found:
                    raise ValueError('Cannot find page template called %s' % ptn)
            if not c:
                raise ValueError('No valid page templates in cycle')
            elif c._restart > len(c):
                raise ValueError('Invalid cycle restart position')
            self._nextPageTemplateCycle = c
        else:
            raise TypeError('argument pt should be string or integer or list')

    def _peekNextPageTemplate(self, pt):
        if isinstance(pt, strTypes):
            for t in self.pageTemplates:
                if t.id == pt:
                    return t
            raise ValueError("can't find template('%s')" % pt)
        elif isinstance(pt, int):
            self.pageTemplates[pt]
        elif isSeq(pt):
            c = PTCycle()
            for ptn in pt:
                found = 0
                if ptn == '*':
                    c._restart = len(c)
                    continue
                for t in self.pageTemplates:
                    if t.id == ptn:
                        c.append(t)
                        found = 1
                if not found:
                    raise ValueError('Cannot find page template called %s' % ptn)
            if not c:
                raise ValueError('No valid page templates in cycle')
            elif c._restart > len(c):
                raise ValueError('Invalid cycle restart position')
            return c.peek
        else:
            raise TypeError('argument pt should be string or integer or list')

    def _peekNextFrame(self):
        """intended to be used by extreme flowables"""
        if hasattr(self, '_nextFrameIndex'):
            return self.pageTemplate.frames[self._nextFrameIndex]
        f = self.frame
        if hasattr(f, 'lastFrame') or f is self.pageTemplate.frames[-1]:
            if hasattr(self, '_nextPageTemplateCycle'):
                pageTemplate = self._nextPageTemplateCycle.peek
            elif hasattr(self, '_nextPageTemplateIndex'):
                pageTemplate = self.pageTemplates[self._nextPageTemplateIndex]
            elif self.pageTemplate.autoNextPageTemplate:
                pageTemplate = self._peekNextPageTemplate(self.pageTemplate.autoNextPageTemplate)
            else:
                pageTemplate = self.pageTemplate
            return pageTemplate.frames[0]
        else:
            return self.pageTemplate.frames[self.pageTemplate.frames.index(f) + 1]

    def handle_nextFrame(self, fx, resume=0):
        """On endFrame change to the frame with name or index fx"""
        if isinstance(fx, strTypes):
            for f in self.pageTemplate.frames:
                if f.id == fx:
                    self._nextFrameIndex = self.pageTemplate.frames.index(f)
                    return
            raise ValueError("can't find frame('%s') in %r(%s) which has frames %r" % (fx, self.pageTemplate, self.pageTemplate.id, [(f, f.id) for f in self.pageTemplate.frames]))
        elif isinstance(fx, int):
            self._nextFrameIndex = fx
        else:
            raise TypeError('argument fx should be string or integer')

    def handle_currentFrame(self, fx, resume=0):
        """change to the frame with name or index fx"""
        self.handle_nextFrame(fx, resume)
        self.handle_frameEnd(resume)

    def handle_breakBefore(self, flowables):
        """preprocessing step to allow pageBreakBefore and frameBreakBefore attributes"""
        first = flowables[0]
        if hasattr(first, '_skipMeNextTime'):
            delattr(first, '_skipMeNextTime')
            return
        if hasattr(first, 'pageBreakBefore') and first.pageBreakBefore == 1:
            first._skipMeNextTime = 1
            first.insert(0, PageBreak())
            return
        if hasattr(first, 'style') and hasattr(first.style, 'pageBreakBefore') and (first.style.pageBreakBefore == 1):
            first._skipMeNextTime = 1
            flowables.insert(0, PageBreak())
            return
        if hasattr(first, 'frameBreakBefore') and first.frameBreakBefore == 1:
            first._skipMeNextTime = 1
            flowables.insert(0, FrameBreak())
            return
        if hasattr(first, 'style') and hasattr(first.style, 'frameBreakBefore') and (first.style.frameBreakBefore == 1):
            first._skipMeNextTime = 1
            flowables.insert(0, FrameBreak())
            return

    def handle_keepWithNext(self, flowables):
        """implements keepWithNext"""
        i = 0
        n = len(flowables)
        while i < n and flowables[i].getKeepWithNext() and _ktAllow(flowables[i]):
            i += 1
        if i:
            if i < n and _ktAllow(flowables[i]):
                i += 1
            K = self.keepTogetherClass(flowables[:i])
            mbe = getattr(self, '_multiBuildEdits', None)
            if mbe:
                for f in K._content[:-1]:
                    if hasattr(f, 'keepWithNext'):
                        mbe((setattr, f, 'keepWithNext', f.keepWithNext))
                    else:
                        mbe((delattr, f, 'keepWithNext'))
                    f.__dict__['keepWithNext'] = 0
            else:
                for f in K._content[:-1]:
                    f.__dict__['keepWithNext'] = 0
            del flowables[:i]
            flowables.insert(0, K)

    def _fIdent(self, f, maxLen=None, frame=None):
        if frame:
            f._frame = frame
        try:
            return f.identity(maxLen)
        finally:
            if frame:
                del f._frame

    def handle_flowable(self, flowables):
        """try to handle one flowable from the front of list flowables."""
        self.filterFlowables(flowables)
        f = flowables[0]
        if f:
            self.handle_breakBefore(flowables)
            self.handle_keepWithNext(flowables)
            f = flowables[0]
        del flowables[0]
        if f is None:
            return
        if isinstance(f, PageBreak):
            npt = f.nextTemplate
            if npt and (not self._samePT(npt)):
                npt = NextPageTemplate(npt)
                npt.apply(self)
                self.afterFlowable(npt)
            if isinstance(f, SlowPageBreak):
                self.handle_pageBreak(slow=1)
            else:
                self.handle_pageBreak()
            self.afterFlowable(f)
        elif isinstance(f, ActionFlowable):
            f.apply(self)
            self.afterFlowable(f)
        else:
            frame = self.frame
            canv = self.canv
            if frame.add(f, canv, trySplit=self.allowSplitting):
                if not isinstance(f, FrameActionFlowable):
                    self._curPageFlowableCount += 1
                    self.afterFlowable(f)
                _addGeneratedContent(flowables, frame)
            else:
                if self.allowSplitting:
                    S = frame.split(f, canv)
                    n = len(S)
                else:
                    n = 0
                if n:
                    if not isinstance(S[0], (PageBreak, SlowPageBreak, ActionFlowable, DDIndenter)):
                        if not frame.add(S[0], canv, trySplit=0):
                            ident = 'Splitting error(n==%d) on page %d in\n%s\nS[0]=%s' % (n, self.page, self._fIdent(f, 60, frame), self._fIdent(S[0], 60, frame))
                            raise LayoutError(ident)
                        self._curPageFlowableCount += 1
                        self.afterFlowable(S[0])
                        flowables[0:0] = S[1:]
                        _addGeneratedContent(flowables, frame)
                    else:
                        flowables[0:0] = S
                else:
                    if hasattr(f, '_postponed'):
                        ident = 'Flowable %s%s too large on page %d in frame %r%s of template %r' % (self._fIdent(f, 60, frame), _fSizeString(f), self.page, self.frame.id, self.frame._aSpaceString(), self.pageTemplate.id)
                        raise LayoutError(ident)
                    f._postponed = 1
                    mbe = getattr(self, '_multiBuildEdits', None)
                    if mbe:
                        mbe((delattr, f, '_postponed'))
                    flowables.insert(0, f)
                    self.handle_frameEnd()
    _handle_documentBegin = handle_documentBegin
    _handle_pageBegin = handle_pageBegin
    _handle_pageEnd = handle_pageEnd
    _handle_frameBegin = handle_frameBegin
    _handle_frameEnd = handle_frameEnd
    _handle_flowable = handle_flowable
    _handle_nextPageTemplate = handle_nextPageTemplate
    _handle_currentFrame = handle_currentFrame
    _handle_nextFrame = handle_nextFrame

    def _makeCanvas(self, filename=None, canvasmaker=canvas.Canvas):
        """make and return a sample canvas. As suggested by 
        Chris Jerdonek cjerdonek @ bitbucket this allows testing of stringWidths
        etc.

        *NB* only the canvases created in self._startBuild will actually be used
        in the build process.
        """
        self.seq = reportlab.lib.sequencer.Sequencer()
        canv = canvasmaker(filename or self.filename, pagesize=self.pagesize, invariant=self.invariant, pageCompression=self.pageCompression, enforceColorSpace=self.enforceColorSpace, initialFontName=self.initialFontName, initialFontSize=self.initialFontSize, initialLeading=self.initialLeading, cropBox=self.cropBox, artBox=self.artBox, trimBox=self.trimBox, bleedBox=self.bleedBox, lang=self.lang)
        getattr(canv, 'setEncrypt', lambda x: None)(self.encrypt)
        canv._cropMarks = self.cropMarks
        canv.setAuthor(self.author)
        canv.setTitle(self.title)
        canv.setSubject(self.subject)
        canv.setCreator(self.creator)
        canv.setProducer(self.producer)
        canv.setKeywords(self.keywords)
        from reportlab.pdfbase.pdfdoc import ViewerPreferencesPDFDictionary as VPD, checkPDFBoolean as cPDFB
        for k, vf in VPD.validate.items():
            v = getattr(self, k[0].lower() + k[1:], None)
            if v is not None:
                if vf is cPDFB:
                    v = ['false', 'true'][v]
                canv.setViewerPreference(k, v)
        if self._onPage:
            canv.setPageCallBack(self._onPage)
        return canv

    def _startBuild(self, filename=None, canvasmaker=canvas.Canvas):
        self._calc()
        self.canv = self._makeCanvas(filename=filename, canvasmaker=canvasmaker)
        self.handle_documentBegin()

    def _endBuild(self):
        self._removeVars(('build', 'page', 'frame'))
        if self._hanging != [] and self._hanging[-1] is PageBegin:
            del self._hanging[-1]
            self.clean_hanging()
        else:
            self.clean_hanging()
            self.handle_pageBreak()
        if getattr(self, '_doSave', 1):
            self.canv.save()
        if self._onPage:
            self.canv.setPageCallBack(None)

    def build(self, flowables, filename=None, canvasmaker=canvas.Canvas):
        """Build the document from a list of flowables.
           If the filename argument is provided then that filename is used
           rather than the one provided upon initialization.
           If the canvasmaker argument is provided then it will be used
           instead of the default.  For example a slideshow might use
           an alternate canvas which places 6 slides on a page (by
           doing translations, scalings and redefining the page break
           operations).
        """
        flowableCount = len(flowables)
        if self._onProgress:
            self._onProgress('STARTED', 0)
            self._onProgress('SIZE_EST', len(flowables))
        self._startBuild(filename, canvasmaker)
        canv = self.canv
        self._savedInfo = canv._doc.info
        handled = 0
        try:
            canv._doctemplate = self
            while len(flowables):
                if self._hanging and self._hanging[-1] is PageBegin and isinstance(flowables[0], PageBreakIfNotEmpty):
                    npt = flowables[0].nextTemplate
                    if npt and (not self._samePT(npt)):
                        npt = NextPageTemplate(npt)
                        npt.apply(self)
                        self._setPageTemplate()
                    del flowables[0]
                self.clean_hanging()
                try:
                    first = flowables[0]
                    self.handle_flowable(flowables)
                    handled += 1
                except:
                    if hasattr(first, '_traceInfo') and first._traceInfo:
                        exc = sys.exc_info()[1]
                        args = list(exc.args)
                        tr = first._traceInfo
                        args[0] += '\n(srcFile %s, line %d char %d to line %d char %d)' % (tr.srcFile, tr.startLineNo, tr.startLinePos, tr.endLineNo, tr.endLinePos)
                        exc.args = tuple(args)
                    raise
                if self._onProgress:
                    self._onProgress('PROGRESS', flowableCount - len(flowables))
        finally:
            del canv._doctemplate
        canv._doc.info = self._savedInfo
        self._endBuild()
        if self._onProgress:
            self._onProgress('FINISHED', 0)

    def _allSatisfied(self):
        """Called by multi-build - are all cross-references resolved?"""
        allHappy = 1
        for f in self._indexingFlowables:
            if not f.isSatisfied():
                allHappy = 0
                break
        return allHappy

    def notify(self, kind, stuff):
        """Forward to any listeners"""
        for l in self._indexingFlowables:
            _canv = getattr(l, '_canv', self)
            try:
                if _canv == self:
                    l._canv = self.canv
                l.notify(kind, stuff)
            finally:
                if _canv == self:
                    del l._canv

    def pageRef(self, label):
        """hook to register a page number"""
        if verbose:
            print("pageRef called with label '%s' on page %d" % (label, self.page))
        self._pageRefs[label] = self.page

    def multiBuild(self, story, maxPasses=10, **buildKwds):
        """Makes multiple passes until all indexing flowables
        are happy.

        Returns number of passes"""
        self._indexingFlowables = []
        for thing in story:
            if thing.isIndexing():
                self._indexingFlowables.append(thing)
        self._doSave = 0
        passes = 0
        mbe = []
        self._multiBuildEdits = mbe.append
        while 1:
            passes += 1
            if self._onProgress:
                self._onProgress('PASS', passes)
            if verbose:
                sys.stdout.write('building pass ' + str(passes) + '...')
            for fl in self._indexingFlowables:
                fl.beforeBuild()
            tempStory = story[:]
            self.build(tempStory, **buildKwds)
            for fl in self._indexingFlowables:
                fl.afterBuild()
            happy = self._allSatisfied()
            if happy:
                self._doSave = 0
                self.canv.save()
                break
            if passes > maxPasses:
                raise IndexError('Index entries not resolved after %d passes' % maxPasses)
            while mbe:
                e = mbe.pop(0)
                e[0](*e[1:])
        del self._multiBuildEdits
        if verbose:
            print('saved')
        return passes

    def afterInit(self):
        """This is called after initialisation of the base class."""
        pass

    def beforeDocument(self):
        """This is called before any processing is
        done on the document."""
        pass

    def beforePage(self):
        """This is called at the beginning of page
        processing, and immediately before the
        beforeDrawPage method of the current page
        template."""
        pass

    def afterPage(self):
        """This is called after page processing, and
        immediately after the afterDrawPage method
        of the current page template."""
        pass

    def filterFlowables(self, flowables):
        """called to filter flowables at the start of the main handle_flowable method.
        Upon return if flowables[0] has been set to None it is discarded and the main
        method returns.
        """
        pass

    def afterFlowable(self, flowable):
        """called after a flowable has been rendered"""
        pass
    _allowedLifetimes = ('page', 'frame', 'build', 'forever')

    def docAssign(self, var, expr, lifetime):
        if not isinstance(expr, strTypes):
            expr = str(expr)
        expr = expr.strip()
        var = var.strip()
        self.docExec('%s=(%s)' % (var.strip(), expr.strip()), lifetime)

    def docExec(self, stmt, lifetime):
        stmt = stmt.strip()
        NS = self._nameSpace
        K0 = list(NS.keys())
        try:
            if lifetime not in self._allowedLifetimes:
                raise ValueError('bad lifetime %r not in %r' % (lifetime, self._allowedLifetimes))
            exec(stmt, NS)
        except:
            K1 = [k for k in NS if k not in K0]
            for k in K1:
                del NS[k]
            annotateException('\ndocExec %s lifetime=%r failed!\n' % (stmt, lifetime))
        self._addVars([k for k in NS.keys() if k not in K0], lifetime)

    def _addVars(self, vars, lifetime):
        """add namespace variables to lifetimes lists"""
        LT = self._lifetimes
        for var in vars:
            for v in LT.values():
                if var in v:
                    v.remove(var)
            LT.setdefault(lifetime, set([])).add(var)

    def _removeVars(self, lifetimes):
        """remove namespace variables for with lifetime in lifetimes"""
        LT = self._lifetimes
        NS = self._nameSpace
        for lifetime in lifetimes:
            for k in LT.setdefault(lifetime, []):
                try:
                    del NS[k]
                except KeyError:
                    pass
            del LT[lifetime]

    def docEval(self, expr):
        try:
            return eval(expr.strip(), {}, self._nameSpace)
        except:
            annotateException('\ndocEval %s failed!\n' % expr)