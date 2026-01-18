class OutOfDateTree(BzrError):
    _fmt = "Working tree is out of date, please run 'brz update'.%(more)s"

    def __init__(self, tree, more=None):
        if more is None:
            more = ''
        else:
            more = ' ' + more
        BzrError.__init__(self)
        self.tree = tree
        self.more = more