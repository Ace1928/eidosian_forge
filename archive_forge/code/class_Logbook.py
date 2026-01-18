from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import eq
class Logbook(list):
    """Evolution records as a chronological list of dictionaries.

    Data can be retrieved via the :meth:`select` method given the appropriate
    names.

    The :class:`Logbook` class may also contain other logbooks referred to
    as chapters. Chapters are used to store information associated to a
    specific part of the evolution. For example when computing statistics
    on different components of individuals (namely :class:`MultiStatistics`),
    chapters can be used to distinguish the average fitness and the average
    size.
    """

    def __init__(self):
        self.buffindex = 0
        self.chapters = defaultdict(Logbook)
        'Dictionary containing the sub-sections of the logbook which are also\n        :class:`Logbook`. Chapters are automatically created when the right hand\n        side of a keyworded argument, provided to the *record* function, is a\n        dictionary. The keyword determines the chapter\'s name. For example, the\n        following line adds a new chapter "size" that will contain the fields\n        "max" and "mean". ::\n\n            logbook.record(gen=0, size={\'max\' : 10.0, \'mean\' : 7.5})\n\n        To access a specific chapter, use the name of the chapter as a\n        dictionary key. For example, to access the size chapter and select\n        the mean use ::\n\n            logbook.chapters["size"].select("mean")\n\n        Compiling a :class:`MultiStatistics` object returns a dictionary\n        containing dictionaries, therefore when recording such an object in a\n        logbook using the keyword argument unpacking operator (**), chapters\n        will be automatically added to the logbook.\n        ::\n\n            >>> fit_stats = Statistics(key=attrgetter("fitness.values"))\n            >>> size_stats = Statistics(key=len)\n            >>> mstats = MultiStatistics(fitness=fit_stats, size=size_stats)\n            >>> # [...]\n            >>> record = mstats.compile(population)\n            >>> logbook.record(**record)\n            >>> print logbook\n              fitness          length\n            ------------    ------------\n            max     mean    max     mean\n            2       1       4       3\n\n        '
        self.columns_len = None
        self.header = None
        'Order of the columns to print when using the :data:`stream` and\n        :meth:`__str__` methods. The syntax is a single iterable containing\n        string elements. For example, with the previously\n        defined statistics class, one can print the generation and the\n        fitness average, and maximum with\n        ::\n\n            logbook.header = ("gen", "mean", "max")\n\n        If not set the header is built with all fields, in arbitrary order\n        on insertion of the first data. The header can be removed by setting\n        it to :data:`None`.\n        '
        self.log_header = True
        'Tells the log book to output or not the header when streaming the\n        first line or getting its entire string representation. This defaults\n        :data:`True`.\n        '

    def record(self, **infos):
        """Enter a record of event in the logbook as a list of key-value pairs.
        The information are appended chronologically to a list as a dictionary.
        When the value part of a pair is a dictionary, the information contained
        in the dictionary are recorded in a chapter entitled as the name of the
        key part of the pair. Chapters are also Logbook.
        """
        apply_to_all = {k: v for k, v in infos.items() if not isinstance(v, dict)}
        for key, value in list(infos.items()):
            if isinstance(value, dict):
                chapter_infos = value.copy()
                chapter_infos.update(apply_to_all)
                self.chapters[key].record(**chapter_infos)
                del infos[key]
        self.append(infos)

    def select(self, *names):
        """Return a list of values associated to the *names* provided
        in argument in each dictionary of the Statistics object list.
        One list per name is returned in order.
        ::

            >>> log = Logbook()
            >>> log.record(gen=0, mean=5.4, max=10.0)
            >>> log.record(gen=1, mean=9.4, max=15.0)
            >>> log.select("mean")
            [5.4, 9.4]
            >>> log.select("gen", "max")
            ([0, 1], [10.0, 15.0])

        With a :class:`MultiStatistics` object, the statistics for each
        measurement can be retrieved using the :data:`chapters` member :
        ::

            >>> log = Logbook()
            >>> log.record(**{'gen': 0, 'fit': {'mean': 0.8, 'max': 1.5},
            ... 'size': {'mean': 25.4, 'max': 67}})
            >>> log.record(**{'gen': 1, 'fit': {'mean': 0.95, 'max': 1.7},
            ... 'size': {'mean': 28.1, 'max': 71}})
            >>> log.chapters['size'].select("mean")
            [25.4, 28.1]
            >>> log.chapters['fit'].select("gen", "max")
            ([0, 1], [1.5, 1.7])
        """
        if len(names) == 1:
            return [entry.get(names[0], None) for entry in self]
        return tuple(([entry.get(name, None) for entry in self] for name in names))

    @property
    def stream(self):
        """Retrieve the formatted not streamed yet entries of the database
        including the headers.
        ::

            >>> log = Logbook()
            >>> log.append({'gen' : 0})
            >>> print log.stream  # doctest: +NORMALIZE_WHITESPACE
            gen
            0
            >>> log.append({'gen' : 1})
            >>> print log.stream  # doctest: +NORMALIZE_WHITESPACE
            1
        """
        startindex, self.buffindex = (self.buffindex, len(self))
        return self.__str__(startindex)

    def __delitem__(self, key):
        if isinstance(key, slice):
            for i, in range(*key.indices(len(self))):
                self.pop(i)
                for chapter in self.chapters.values():
                    chapter.pop(i)
        else:
            self.pop(key)
            for chapter in self.chapters.values():
                chapter.pop(key)

    def pop(self, index=0):
        """Retrieve and delete element *index*. The header and stream will be
        adjusted to follow the modification.

        :param item: The index of the element to remove, optional. It defaults
                     to the first element.

        You can also use the following syntax to delete elements.
        ::

            del log[0]
            del log[1::5]
        """
        if index < self.buffindex:
            self.buffindex -= 1
        return super(self.__class__, self).pop(index)

    def __txt__(self, startindex):
        columns = self.header
        if not columns:
            columns = sorted(self[0].keys()) + sorted(self.chapters.keys())
        if not self.columns_len or len(self.columns_len) != len(columns):
            self.columns_len = [len(c) for c in columns]
        chapters_txt = {}
        offsets = defaultdict(int)
        for name, chapter in self.chapters.items():
            chapters_txt[name] = chapter.__txt__(startindex)
            if startindex == 0:
                offsets[name] = len(chapters_txt[name]) - len(self)
        str_matrix = []
        for i, line in enumerate(self[startindex:]):
            str_line = []
            for j, name in enumerate(columns):
                if name in chapters_txt:
                    column = chapters_txt[name][i + offsets[name]]
                else:
                    value = line.get(name, '')
                    string = '{0:n}' if isinstance(value, float) else '{0}'
                    column = string.format(value)
                self.columns_len[j] = max(self.columns_len[j], len(column))
                str_line.append(column)
            str_matrix.append(str_line)
        if startindex == 0 and self.log_header:
            header = []
            nlines = 1
            if len(self.chapters) > 0:
                nlines += max(map(len, chapters_txt.values())) - len(self) + 1
            header = [[] for i in range(nlines)]
            for j, name in enumerate(columns):
                if name in chapters_txt:
                    length = max((len(line.expandtabs()) for line in chapters_txt[name]))
                    blanks = nlines - 2 - offsets[name]
                    for i in range(blanks):
                        header[i].append(' ' * length)
                    header[blanks].append(name.center(length))
                    header[blanks + 1].append('-' * length)
                    for i in range(offsets[name]):
                        header[blanks + 2 + i].append(chapters_txt[name][i])
                else:
                    length = max((len(line[j].expandtabs()) for line in str_matrix))
                    for line in header[:-1]:
                        line.append(' ' * length)
                    header[-1].append(name)
            str_matrix = chain(header, str_matrix)
        template = '\t'.join(('{%i:<%i}' % (i, l) for i, l in enumerate(self.columns_len)))
        text = [template.format(*line) for line in str_matrix]
        return text

    def __str__(self, startindex=0):
        text = self.__txt__(startindex)
        return '\n'.join(text)