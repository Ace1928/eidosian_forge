from qpd_test.tests_base import TestsBase
from typing import Any, Tuple, Iterable
from pandasql import sqldf
from timeit import timeit
def _test_integration_1(self, *sizes: int, repeat: int=1) -> Iterable[Tuple[int, float, float]]:
    for n in sizes:
        a = self.make_rand_df(n, a=int, b=str, c=float, d=int, e=bool, f=str, g=str, h=float)
        q, s = self._time_vs_sqlite(repeat, '\n                    WITH\n                        a1 AS (\n                            SELECT a+1 AS a, b, c FROM a\n                        ),\n                        a2 AS (\n                            SELECT a,MAX(b) AS b_max, AVG(c) AS c_avg FROM a GROUP BY a\n                        ),\n                        a3 AS (\n                            SELECT d+2 AS d, f, g, h FROM a WHERE e\n                        )\n                    SELECT a1.a,b,c,b_max,c_avg,f,g,h FROM a1\n                        INNER JOIN a2 ON a1.a=a2.a\n                        LEFT JOIN a3 ON a1.a=a3.d\n                    ', a=a)
        yield (n, q, s)