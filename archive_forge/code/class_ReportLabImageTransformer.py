class ReportLabImageTransformer(object):

    def __init__(self, smiCol, width=1.0, verbose=1, tempHandler=None):
        self.smiCol = smiCol
        self.width = width * inch
        self.verbose = verbose

    def __call__(self, arg):
        res = list(arg)
        if self.verbose:
            sys.stderr.write('Render(%d): %s\n' % (self.smiCol, str(res[0])))
        smi = res[self.smiCol]
        aspect = 1
        width = self.width
        height = aspect * width
        try:
            mol = Chem.MolFromSmiles(smi)
            Chem.Kekulize(mol)
            canv = Canvas((width, height))
            options = DrawingOptions()
            options.atomLabelMinFontSize = 3
            options.bondLineWidth = 0.5
            drawing = MolDrawing(options=options)
            if not mol.GetNumConformers():
                rdDepictor.Compute2DCoords(mol)
            drawing.AddMol(mol, canvas=canv)
            ok = True
        except Exception:
            if self.verbose:
                import traceback
                traceback.print_exc()
            ok = False
        if ok:
            res[self.smiCol] = canv.drawing
        else:
            res[self.smiCol] = 'Failed'
        return res