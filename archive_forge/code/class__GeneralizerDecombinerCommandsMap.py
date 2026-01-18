from fontTools.cffLib import maxStackLimit
class _GeneralizerDecombinerCommandsMap(object):

    @staticmethod
    def rmoveto(args):
        if len(args) != 2:
            raise ValueError(args)
        yield ('rmoveto', args)

    @staticmethod
    def hmoveto(args):
        if len(args) != 1:
            raise ValueError(args)
        yield ('rmoveto', [args[0], 0])

    @staticmethod
    def vmoveto(args):
        if len(args) != 1:
            raise ValueError(args)
        yield ('rmoveto', [0, args[0]])

    @staticmethod
    def rlineto(args):
        if not args:
            raise ValueError(args)
        for args in _everyN(args, 2):
            yield ('rlineto', args)

    @staticmethod
    def hlineto(args):
        if not args:
            raise ValueError(args)
        it = iter(args)
        try:
            while True:
                yield ('rlineto', [next(it), 0])
                yield ('rlineto', [0, next(it)])
        except StopIteration:
            pass

    @staticmethod
    def vlineto(args):
        if not args:
            raise ValueError(args)
        it = iter(args)
        try:
            while True:
                yield ('rlineto', [0, next(it)])
                yield ('rlineto', [next(it), 0])
        except StopIteration:
            pass

    @staticmethod
    def rrcurveto(args):
        if not args:
            raise ValueError(args)
        for args in _everyN(args, 6):
            yield ('rrcurveto', args)

    @staticmethod
    def hhcurveto(args):
        if len(args) < 4 or len(args) % 4 > 1:
            raise ValueError(args)
        if len(args) % 2 == 1:
            yield ('rrcurveto', [args[1], args[0], args[2], args[3], args[4], 0])
            args = args[5:]
        for args in _everyN(args, 4):
            yield ('rrcurveto', [args[0], 0, args[1], args[2], args[3], 0])

    @staticmethod
    def vvcurveto(args):
        if len(args) < 4 or len(args) % 4 > 1:
            raise ValueError(args)
        if len(args) % 2 == 1:
            yield ('rrcurveto', [args[0], args[1], args[2], args[3], 0, args[4]])
            args = args[5:]
        for args in _everyN(args, 4):
            yield ('rrcurveto', [0, args[0], args[1], args[2], 0, args[3]])

    @staticmethod
    def hvcurveto(args):
        if len(args) < 4 or len(args) % 8 not in {0, 1, 4, 5}:
            raise ValueError(args)
        last_args = None
        if len(args) % 2 == 1:
            lastStraight = len(args) % 8 == 5
            args, last_args = (args[:-5], args[-5:])
        it = _everyN(args, 4)
        try:
            while True:
                args = next(it)
                yield ('rrcurveto', [args[0], 0, args[1], args[2], 0, args[3]])
                args = next(it)
                yield ('rrcurveto', [0, args[0], args[1], args[2], args[3], 0])
        except StopIteration:
            pass
        if last_args:
            args = last_args
            if lastStraight:
                yield ('rrcurveto', [args[0], 0, args[1], args[2], args[4], args[3]])
            else:
                yield ('rrcurveto', [0, args[0], args[1], args[2], args[3], args[4]])

    @staticmethod
    def vhcurveto(args):
        if len(args) < 4 or len(args) % 8 not in {0, 1, 4, 5}:
            raise ValueError(args)
        last_args = None
        if len(args) % 2 == 1:
            lastStraight = len(args) % 8 == 5
            args, last_args = (args[:-5], args[-5:])
        it = _everyN(args, 4)
        try:
            while True:
                args = next(it)
                yield ('rrcurveto', [0, args[0], args[1], args[2], args[3], 0])
                args = next(it)
                yield ('rrcurveto', [args[0], 0, args[1], args[2], 0, args[3]])
        except StopIteration:
            pass
        if last_args:
            args = last_args
            if lastStraight:
                yield ('rrcurveto', [0, args[0], args[1], args[2], args[3], args[4]])
            else:
                yield ('rrcurveto', [args[0], 0, args[1], args[2], args[4], args[3]])

    @staticmethod
    def rcurveline(args):
        if len(args) < 8 or len(args) % 6 != 2:
            raise ValueError(args)
        args, last_args = (args[:-2], args[-2:])
        for args in _everyN(args, 6):
            yield ('rrcurveto', args)
        yield ('rlineto', last_args)

    @staticmethod
    def rlinecurve(args):
        if len(args) < 8 or len(args) % 2 != 0:
            raise ValueError(args)
        args, last_args = (args[:-6], args[-6:])
        for args in _everyN(args, 2):
            yield ('rlineto', args)
        yield ('rrcurveto', last_args)