class NDArrayMeta(type):

    def __getitem__(cls, item):
        return NDArray(item)