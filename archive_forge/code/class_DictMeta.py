class DictMeta(type):

    def __getitem__(cls, item):
        return Dict(item)