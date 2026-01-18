from .psycopg2 import PGDialect_psycopg2
from ... import util
@util.memoized_property
def _psycopg2_extensions(cls):
    root = __import__('psycopg2cffi', fromlist=['extensions'])
    return root.extensions