import os
import re
from collections import defaultdict
from itertools import chain
from nltk.corpus.reader.util import *
from nltk.data import FileSystemPathPointer, PathPointer, ZipFilePathPointer
class CategorizedCorpusReader:
    """
    A mixin class used to aid in the implementation of corpus readers
    for categorized corpora.  This class defines the method
    ``categories()``, which returns a list of the categories for the
    corpus or for a specified set of fileids; and overrides ``fileids()``
    to take a ``categories`` argument, restricting the set of fileids to
    be returned.

    Subclasses are expected to:

      - Call ``__init__()`` to set up the mapping.

      - Override all view methods to accept a ``categories`` parameter,
        which can be used *instead* of the ``fileids`` parameter, to
        select which fileids should be included in the returned view.
    """

    def __init__(self, kwargs):
        """
        Initialize this mapping based on keyword arguments, as
        follows:

          - cat_pattern: A regular expression pattern used to find the
            category for each file identifier.  The pattern will be
            applied to each file identifier, and the first matching
            group will be used as the category label for that file.

          - cat_map: A dictionary, mapping from file identifiers to
            category labels.

          - cat_file: The name of a file that contains the mapping
            from file identifiers to categories.  The argument
            ``cat_delimiter`` can be used to specify a delimiter.

        The corresponding argument will be deleted from ``kwargs``.  If
        more than one argument is specified, an exception will be
        raised.
        """
        self._f2c = None
        self._c2f = None
        self._pattern = None
        self._map = None
        self._file = None
        self._delimiter = None
        if 'cat_pattern' in kwargs:
            self._pattern = kwargs['cat_pattern']
            del kwargs['cat_pattern']
        elif 'cat_map' in kwargs:
            self._map = kwargs['cat_map']
            del kwargs['cat_map']
        elif 'cat_file' in kwargs:
            self._file = kwargs['cat_file']
            del kwargs['cat_file']
            if 'cat_delimiter' in kwargs:
                self._delimiter = kwargs['cat_delimiter']
                del kwargs['cat_delimiter']
        else:
            raise ValueError('Expected keyword argument cat_pattern or cat_map or cat_file.')
        if 'cat_pattern' in kwargs or 'cat_map' in kwargs or 'cat_file' in kwargs:
            raise ValueError('Specify exactly one of: cat_pattern, cat_map, cat_file.')

    def _init(self):
        self._f2c = defaultdict(set)
        self._c2f = defaultdict(set)
        if self._pattern is not None:
            for file_id in self._fileids:
                category = re.match(self._pattern, file_id).group(1)
                self._add(file_id, category)
        elif self._map is not None:
            for file_id, categories in self._map.items():
                for category in categories:
                    self._add(file_id, category)
        elif self._file is not None:
            with self.open(self._file) as f:
                for line in f.readlines():
                    line = line.strip()
                    file_id, categories = line.split(self._delimiter, 1)
                    if file_id not in self.fileids():
                        raise ValueError('In category mapping file %s: %s not found' % (self._file, file_id))
                    for category in categories.split(self._delimiter):
                        self._add(file_id, category)

    def _add(self, file_id, category):
        self._f2c[file_id].add(category)
        self._c2f[category].add(file_id)

    def categories(self, fileids=None):
        """
        Return a list of the categories that are defined for this corpus,
        or for the file(s) if it is given.
        """
        if self._f2c is None:
            self._init()
        if fileids is None:
            return sorted(self._c2f)
        if isinstance(fileids, str):
            fileids = [fileids]
        return sorted(set.union(*(self._f2c[d] for d in fileids)))

    def fileids(self, categories=None):
        """
        Return a list of file identifiers for the files that make up
        this corpus, or that make up the given category(s) if specified.
        """
        if categories is None:
            return super().fileids()
        elif isinstance(categories, str):
            if self._f2c is None:
                self._init()
            if categories in self._c2f:
                return sorted(self._c2f[categories])
            else:
                raise ValueError('Category %s not found' % categories)
        else:
            if self._f2c is None:
                self._init()
            return sorted(set.union(*(self._c2f[c] for c in categories)))

    def _resolve(self, fileids, categories):
        if fileids is not None and categories is not None:
            raise ValueError('Specify fileids or categories, not both')
        if categories is not None:
            return self.fileids(categories)
        else:
            return fileids

    def raw(self, fileids=None, categories=None):
        return super().raw(self._resolve(fileids, categories))

    def words(self, fileids=None, categories=None):
        return super().words(self._resolve(fileids, categories))

    def sents(self, fileids=None, categories=None):
        return super().sents(self._resolve(fileids, categories))

    def paras(self, fileids=None, categories=None):
        return super().paras(self._resolve(fileids, categories))