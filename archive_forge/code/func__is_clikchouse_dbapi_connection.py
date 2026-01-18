from __future__ import absolute_import, print_function, division
import logging
from petl.compat import callable
def _is_clikchouse_dbapi_connection(dbo):
    return 'clickhouse_driver' in str(type(dbo))