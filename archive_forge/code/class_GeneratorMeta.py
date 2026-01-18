class GeneratorMeta(type):

    def __getitem__(cls, item):
        return Generator(item)