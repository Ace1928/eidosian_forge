from __future__ import absolute_import, print_function, division
import operator
from petl.compat import string_types, izip
from petl.errors import ArgumentError
from petl.util.base import Table, dicts
class SearchTextIndexView(Table):

    def __init__(self, index_or_dirname, query, limit=None, pagenum=None, pagelen=None, indexname=None, docnum_field=None, score_field=None, fieldboosts=None, search_kwargs=None):
        self._index_or_dirname = index_or_dirname
        self._query = query
        self._limit = limit
        self._pagenum = pagenum
        self._pagelen = pagelen
        self._indexname = indexname
        self._docnum_field = docnum_field
        self._score_field = score_field
        self._fieldboosts = fieldboosts
        self._search_kwargs = search_kwargs

    def __iter__(self):
        return itersearchindex(self._index_or_dirname, self._query, self._limit, self._pagenum, self._pagelen, self._indexname, self._docnum_field, self._score_field, self._fieldboosts, self._search_kwargs)