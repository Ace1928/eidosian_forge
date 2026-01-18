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
def _mssql_drop_ignore(conn, ident):
    try:
        conn.exec_driver_sql('drop database %s' % ident)
        log.info('Reaped db: %s', ident)
        return True
    except exc.DatabaseError as err:
        log.warning("couldn't drop db: %s", err)
        return False