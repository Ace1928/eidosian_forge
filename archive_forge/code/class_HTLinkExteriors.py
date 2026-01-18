from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class HTLinkExteriors(LinkExteriorsTable):
    """
        Iterator for all knots up to 14 or 15 crossings (see below for
        which) and links up to 14 crossings as tabulated by Jim Hoste
        and Morwen Thistlethwaite.  In addition to the filter
        arguments supported by all ManifoldTables, this iterator
        provides alternating=<True/False>;
        knots_vs_links=<'knots'/'links'>; and crossings=N. These allow
        iterations only through alternating or non-alternating links
        with 1 or more than 1 component and a specified crossing
        number.

        >>> HTLinkExteriors.identify(LinkExteriors['8_20'])
        K8n1(0,0)
        >>> Mylist = HTLinkExteriors(alternating=False,knots_vs_links='links')[8.5:8.7]
        >>> len(Mylist)
        8
        >>> for L in Mylist: # doctest: +NUMERIC6
        ...   print( L.name(), L.num_cusps(), L.volume() )
        ... 
        L11n138 2 8.66421454
        L12n1097 2 8.51918360
        L14n13364 2 8.69338342
        L14n13513 2 8.58439465
        L14n15042 2 8.66421454
        L14n24425 2 8.60676092
        L14n24777 2 8.53123093
        L14n26042 2 8.64333782
        >>> for L in Mylist:
        ...   print( L.name(), L.DT_code() )
        ... 
        L11n138 [(8, -10, -12), (6, -16, -18, -22, -20, -2, -4, -14)]
        L12n1097 [(10, 12, -14, -18), (22, 2, -20, 24, -6, -8, 4, 16)]
        L14n13364 [(8, -10, 12), (6, -18, 20, -22, -26, -24, 2, -4, -28, -16, -14)]
        L14n13513 [(8, -10, 12), (6, -20, 18, -26, -24, -4, 2, -28, -16, -14, -22)]
        L14n15042 [(8, -10, 14), (12, -16, 18, -22, 24, 2, 26, 28, 6, -4, 20)]
        L14n24425 [(10, -12, 14, -16), (-18, 26, -24, 22, -20, -28, -6, 4, -2, 8)]
        L14n24777 [(10, 12, -14, -18), (2, 28, -22, 24, -6, 26, -8, 4, 16, 20)]
        L14n26042 [(10, 12, 14, -20), (8, 2, 28, -22, -24, -26, -6, -16, -18, 4)]

        SnapPy comes with one of two versions of HTLinkExteriors.  The
        smaller original one provides knots and links up to 14
        crossings; the larger adds to that the knots (but not links)
        with 15 crossings.  You can determine which you have by whether

        >>> len(HTLinkExteriors(crossings=15))   # doctest: +SKIP

        gives 0 or 253293. To upgrade to the larger database, install
        the Python module 'snappy_15_knots' as discussed on the
        'installing SnapPy' webpage.
        """
    _regex = re.compile('[KL][0-9]+[an]([0-9]+)$')

    def __init__(self, **kwargs):
        return LinkExteriorsTable.__init__(self, table='HT_links_view', db_path=alt_database_path, **kwargs)

    def _configure(self, **kwargs):
        """
            Process the ManifoldTable filter arguments and then add
            the ones which are specific to links.
            """
        ManifoldTable._configure(self, **kwargs)
        conditions = []
        alt = kwargs.get('alternating', None)
        if alt == True:
            conditions.append("name like '%a%'")
        elif alt == False:
            conditions.append("name like '%n%'")
        flavor = kwargs.get('knots_vs_links', None)
        if flavor == 'knots':
            conditions.append('cusps=1')
        elif flavor == 'links':
            conditions.append('cusps>1')
        if 'crossings' in kwargs:
            N = int(kwargs['crossings'])
            conditions.append("(name like '_%da%%' or name like '_%dn%%')" % (N, N))
        if self._filter:
            if len(conditions) > 0:
                self._filter += ' and ' + ' and '.join(conditions)
        else:
            self._filter = ' and '.join(conditions)