class CactvsImageTransformer(object):

    def __init__(self, smiCol, width=1.0, verbose=1, tempHandler=None):
        self.smiCol = smiCol
        if tempHandler is None:
            tempHandler = ReportUtils.TempFileHandler()
        self.tempHandler = tempHandler
        self.width = width * inch
        self.verbose = verbose

    def __call__(self, arg):
        res = list(arg)
        if self.verbose:
            sys.stderr.write('Render(%d): %s\n' % (self.smiCol, str(res[0])))
        smi = res[self.smiCol]
        tmpName = self.tempHandler.get('.gif')
        aspect = 1
        width = 300
        height = aspect * width
        ok = cactvs.SmilesToGif(smi, tmpName, (width, height))
        if ok:
            try:
                img = platypus.Image(tmpName)
                img.drawWidth = self.width
                img.drawHeight = aspect * self.width
            except Exception:
                ok = 0
        if ok:
            res[self.smiCol] = img
        else:
            res[self.smiCol] = 'Failed'
        return res