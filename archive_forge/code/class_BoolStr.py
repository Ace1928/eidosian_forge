class BoolStr(Formatter):

    @classmethod
    def deserialize(cls, value):
        """Convert a boolean string to a boolean"""
        expr = str(value).lower()
        if 'true' == expr:
            return True
        elif 'false' == expr:
            return False
        else:
            raise ValueError('Unable to deserialize boolean string: %s' % value)