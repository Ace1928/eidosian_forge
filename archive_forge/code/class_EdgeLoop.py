from snappy.snap import t3mlite as t3m
class EdgeLoop(SubcomplexBase):

    def __init__(self, tet_and_perm, edge_index):
        super(TruncatedComplex.EdgeLoop, self).__init__('edgeLoop', tet_and_perm)
        self.edge_index = edge_index

    def tet_and_perm_of_end(self):
        return self.tet_and_perm

    def __repr__(self):
        return 'TruncatedComplex.EdgeLoop(%r, %d)' % (self.tet_and_perm, self.edge_index)