from __future__ import print_function
import sys, sqlite3, re, os, random
from .sqlite_files import __path__ as manifolds_paths
class LinkExteriors(LinkExteriorsTable):
    """
        Iterator for all knots with at most 11 crossings and links with
        at most 10 crossings, using the Rolfsen notation.  The triangulations
        were computed by Joe Christy.

        >>> for K in LinkExteriors(num_cusps=3)[-3:]: # doctest: +NUMERIC6
        ...   print(K, K.volume())
        ... 
        10^3_72(0,0)(0,0)(0,0) 14.35768903
        10^3_73(0,0)(0,0)(0,0) 15.86374431
        10^3_74(0,0)(0,0)(0,0) 15.55091438
        >>> M = Manifold('8_4')
        >>> OrientableCuspedCensus.identify(M)
        s862(0,0)

        By default, the 'identify' returns the first isometric manifold it finds;
        if the optional 'extends_to_link' flag is set, it insists that meridians
        are taken to meridians.

        >>> M = Manifold('7^2_8')
        >>> LinkExteriors.identify(M)
        5^2_1(0,0)(0,0)
        >>> LinkExteriors.identify(M, extends_to_link=True)
        7^2_8(0,0)(0,0)
        """
    _regex = re.compile('([0-9]+_[0-9]+)$|[0-9]+[\\^][0-9]+_[0-9]+$')

    def __init__(self, **kwargs):
        return ManifoldTable.__init__(self, table='link_exteriors_view', db_path=database_path, **kwargs)

    def __call__(self, *args, **kwargs):
        if args:
            if not isinstance(args[0], int) or len(args) > 1:
                raise TypeError('Invalid specification for num_cusps.')
            if not 'num_cusps' in kwargs:
                kwargs['num_cusps'] = args[0]
        return self.__class__(**kwargs)

    def _configure(self, **kwargs):
        """
            Process the ManifoldTable filter arguments and then add
            the ones which are specific to links.
            """
        ManifoldTable._configure(self, **kwargs)
        conditions = []
        flavor = kwargs.get('knots_vs_links', None)
        if flavor == 'knots':
            conditions.append('cusps=1')
        elif flavor == 'links':
            conditions.append('cusps>1')
        if 'crossings' in kwargs:
            N = int(kwargs['crossings'])
            conditions.append("(name like '%d^%%' or name like '%d|_%%' escape '|')" % (N, N))
        if self._filter:
            if len(conditions) > 0:
                self._filter += ' and ' + ' and '.join(conditions)
        else:
            self._filter = ' and '.join(conditions)