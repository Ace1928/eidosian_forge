from unittest import TestCase
import ibis
from fugue import ExecutionEngine, FugueWorkflow, register_default_sql_engine
from fugue_ibis import IbisEngine, as_fugue, as_ibis, run_ibis
def _test1(con: ibis.BaseBackend) -> ibis.Expr:
    tb = con.table('a')
    return tb