class OptionalMeta(type):

    def __getitem__(cls, item):
        return Optional(item)