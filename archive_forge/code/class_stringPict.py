from .pretty_symbology import hobj, vobj, xsym, xobj, pretty_use_unicode, line_width
from sympy.utilities.exceptions import sympy_deprecation_warning
class stringPict:
    """An ASCII picture.
    The pictures are represented as a list of equal length strings.
    """
    LINE = 'line'

    def __init__(self, s, baseline=0):
        """Initialize from string.
        Multiline strings are centered.
        """
        self.s = s
        self.picture = stringPict.equalLengths(s.splitlines())
        self.baseline = baseline
        self.binding = None

    @staticmethod
    def equalLengths(lines):
        if not lines:
            return ['']
        width = max((line_width(line) for line in lines))
        return [line.center(width) for line in lines]

    def height(self):
        """The height of the picture in characters."""
        return len(self.picture)

    def width(self):
        """The width of the picture in characters."""
        return line_width(self.picture[0])

    @staticmethod
    def next(*args):
        """Put a string of stringPicts next to each other.
        Returns string, baseline arguments for stringPict.
        """
        objects = []
        for arg in args:
            if isinstance(arg, str):
                arg = stringPict(arg)
            objects.append(arg)
        newBaseline = max((obj.baseline for obj in objects))
        newHeightBelowBaseline = max((obj.height() - obj.baseline for obj in objects))
        newHeight = newBaseline + newHeightBelowBaseline
        pictures = []
        for obj in objects:
            oneEmptyLine = [' ' * obj.width()]
            basePadding = newBaseline - obj.baseline
            totalPadding = newHeight - obj.height()
            pictures.append(oneEmptyLine * basePadding + obj.picture + oneEmptyLine * (totalPadding - basePadding))
        result = [''.join(lines) for lines in zip(*pictures)]
        return ('\n'.join(result), newBaseline)

    def right(self, *args):
        """Put pictures next to this one.
        Returns string, baseline arguments for stringPict.
        (Multiline) strings are allowed, and are given a baseline of 0.

        Examples
        ========

        >>> from sympy.printing.pretty.stringpict import stringPict
        >>> print(stringPict("10").right(" + ",stringPict("1\\r-\\r2",1))[0])
             1
        10 + -
             2

        """
        return stringPict.next(self, *args)

    def left(self, *args):
        """Put pictures (left to right) at left.
        Returns string, baseline arguments for stringPict.
        """
        return stringPict.next(*args + (self,))

    @staticmethod
    def stack(*args):
        """Put pictures on top of each other,
        from top to bottom.
        Returns string, baseline arguments for stringPict.
        The baseline is the baseline of the second picture.
        Everything is centered.
        Baseline is the baseline of the second picture.
        Strings are allowed.
        The special value stringPict.LINE is a row of '-' extended to the width.
        """
        objects = []
        for arg in args:
            if arg is not stringPict.LINE and isinstance(arg, str):
                arg = stringPict(arg)
            objects.append(arg)
        newWidth = max((obj.width() for obj in objects if obj is not stringPict.LINE))
        lineObj = stringPict(hobj('-', newWidth))
        for i, obj in enumerate(objects):
            if obj is stringPict.LINE:
                objects[i] = lineObj
        newPicture = []
        for obj in objects:
            newPicture.extend(obj.picture)
        newPicture = [line.center(newWidth) for line in newPicture]
        newBaseline = objects[0].height() + objects[1].baseline
        return ('\n'.join(newPicture), newBaseline)

    def below(self, *args):
        """Put pictures under this picture.
        Returns string, baseline arguments for stringPict.
        Baseline is baseline of top picture

        Examples
        ========

        >>> from sympy.printing.pretty.stringpict import stringPict
        >>> print(stringPict("x+3").below(
        ...       stringPict.LINE, '3')[0]) #doctest: +NORMALIZE_WHITESPACE
        x+3
        ---
         3

        """
        s, baseline = stringPict.stack(self, *args)
        return (s, self.baseline)

    def above(self, *args):
        """Put pictures above this picture.
        Returns string, baseline arguments for stringPict.
        Baseline is baseline of bottom picture.
        """
        string, baseline = stringPict.stack(*args + (self,))
        baseline = len(string.splitlines()) - self.height() + self.baseline
        return (string, baseline)

    def parens(self, left='(', right=')', ifascii_nougly=False):
        """Put parentheses around self.
        Returns string, baseline arguments for stringPict.

        left or right can be None or empty string which means 'no paren from
        that side'
        """
        h = self.height()
        b = self.baseline
        if ifascii_nougly and (not pretty_use_unicode()):
            h = 1
            b = 0
        res = self
        if left:
            lparen = stringPict(vobj(left, h), baseline=b)
            res = stringPict(*lparen.right(self))
        if right:
            rparen = stringPict(vobj(right, h), baseline=b)
            res = stringPict(*res.right(rparen))
        return ('\n'.join(res.picture), res.baseline)

    def leftslash(self):
        """Precede object by a slash of the proper size.
        """
        height = max(self.baseline, self.height() - 1 - self.baseline) * 2 + 1
        slash = '\n'.join((' ' * (height - i - 1) + xobj('/', 1) + ' ' * i for i in range(height)))
        return self.left(stringPict(slash, height // 2))

    def root(self, n=None):
        """Produce a nice root symbol.
        Produces ugly results for big n inserts.
        """
        result = self.above('_' * self.width())
        height = self.height()
        slash = '\n'.join((' ' * (height - i - 1) + '/' + ' ' * i for i in range(height)))
        slash = stringPict(slash, height - 1)
        if height > 2:
            downline = stringPict('\\ \n \\', 1)
        else:
            downline = stringPict('\\')
        if n is not None and n.width() > downline.width():
            downline = downline.left(' ' * (n.width() - downline.width()))
            downline = downline.above(n)
        root = downline.right(slash)
        root.baseline = result.baseline - result.height() + root.height()
        return result.left(root)

    def render(self, *args, **kwargs):
        """Return the string form of self.

           Unless the argument line_break is set to False, it will
           break the expression in a form that can be printed
           on the terminal without being broken up.
         """
        if kwargs['wrap_line'] is False:
            return '\n'.join(self.picture)
        if kwargs['num_columns'] is not None:
            ncols = kwargs['num_columns']
        else:
            ncols = self.terminal_width()
        ncols -= 2
        if ncols <= 0:
            ncols = 78
        if self.width() <= ncols:
            return type(self.picture[0])(self)
        i = 0
        svals = []
        do_vspacers = self.height() > 1
        while i < self.width():
            svals.extend([sval[i:i + ncols] for sval in self.picture])
            if do_vspacers:
                svals.append('')
            i += ncols
        if svals[-1] == '':
            del svals[-1]
        return '\n'.join(svals)

    def terminal_width(self):
        """Return the terminal width if possible, otherwise return 0.
        """
        ncols = 0
        try:
            import curses
            import io
            try:
                curses.setupterm()
                ncols = curses.tigetnum('cols')
            except AttributeError:
                from ctypes import windll, create_string_buffer
                h = windll.kernel32.GetStdHandle(-12)
                csbi = create_string_buffer(22)
                res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
                if res:
                    import struct
                    bufx, bufy, curx, cury, wattr, left, top, right, bottom, maxx, maxy = struct.unpack('hhhhHhhhhhh', csbi.raw)
                    ncols = right - left + 1
            except curses.error:
                pass
            except io.UnsupportedOperation:
                pass
        except (ImportError, TypeError):
            pass
        return ncols

    def __eq__(self, o):
        if isinstance(o, str):
            return '\n'.join(self.picture) == o
        elif isinstance(o, stringPict):
            return o.picture == self.picture
        return False

    def __hash__(self):
        return super().__hash__()

    def __str__(self):
        return '\n'.join(self.picture)

    def __repr__(self):
        return 'stringPict(%r,%d)' % ('\n'.join(self.picture), self.baseline)

    def __getitem__(self, index):
        return self.picture[index]

    def __len__(self):
        return len(self.s)