class UnionMeta(type):

    def __getitem__(cls, item):
        return Union(item)