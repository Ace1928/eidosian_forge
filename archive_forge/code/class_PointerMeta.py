class PointerMeta(type):

    def __getitem__(cls, item):
        return Pointer(item)