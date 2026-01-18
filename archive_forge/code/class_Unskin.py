class Unskin(object):

    def __new__(self, *args, **kwargs):
        return args[0].__pts__