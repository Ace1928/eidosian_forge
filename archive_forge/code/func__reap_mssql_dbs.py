from sqlalchemy import inspect
from sqlalchemy import Integer
from ... import create_engine
from ... import exc
from ...schema import Column
from ...schema import DropConstraint
from ...schema import ForeignKeyConstraint
from ...schema import MetaData
from ...schema import Table
from ...testing.provision import create_db
from ...testing.provision import drop_all_schema_objects_pre_tables
from ...testing.provision import drop_db
from ...testing.provision import generate_driver_url
from ...testing.provision import get_temp_table_name
from ...testing.provision import log
from ...testing.provision import normalize_sequence
from ...testing.provision import run_reap_dbs
from ...testing.provision import temp_table_keyword_args
@run_reap_dbs.for_db('mssql')
def _reap_mssql_dbs(url, idents):
    log.info('db reaper connecting to %r', url)
    eng = create_engine(url)
    with eng.connect().execution_options(isolation_level='AUTOCOMMIT') as conn:
        log.info('identifiers in file: %s', ', '.join(idents))
        to_reap = conn.exec_driver_sql("select d.name from sys.databases as d where name like 'TEST_%' and not exists (select session_id from sys.dm_exec_sessions where database_id=d.database_id)")
        all_names = {dbname.lower() for dbname, in to_reap}
        to_drop = set()
        for name in all_names:
            if name in idents:
                to_drop.add(name)
        dropped = total = 0
        for total, dbname in enumerate(to_drop, 1):
            if _mssql_drop_ignore(conn, dbname):
                dropped += 1
        log.info('Dropped %d out of %d stale databases detected', dropped, total)