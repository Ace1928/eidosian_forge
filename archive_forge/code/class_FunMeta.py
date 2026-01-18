class FunMeta(type):

    def __getitem__(cls, item):
        return Fun(tuple(item[0]) + (item[1],))