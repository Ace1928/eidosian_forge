from ... import create_engine
from ... import exc
from ... import inspect
from ...engine import url as sa_url
from ...testing.provision import configure_follower
from ...testing.provision import create_db
from ...testing.provision import drop_all_schema_objects_post_tables
from ...testing.provision import drop_all_schema_objects_pre_tables
from ...testing.provision import drop_db
from ...testing.provision import follower_url_from_main
from ...testing.provision import log
from ...testing.provision import post_configure_engine
from ...testing.provision import run_reap_dbs
from ...testing.provision import set_default_schema_on_connection
from ...testing.provision import stop_test_class_outside_fixtures
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import update_db_opts
@drop_all_schema_objects_post_tables.for_db('oracle')
def _ora_drop_all_schema_objects_post_tables(cfg, eng):
    with eng.begin() as conn:
        for syn in conn.dialect._get_synonyms(conn, None, None, None):
            conn.exec_driver_sql(f'drop synonym {syn['synonym_name']}')
        for syn in conn.dialect._get_synonyms(conn, cfg.test_schema, None, None):
            conn.exec_driver_sql(f'drop synonym {cfg.test_schema}.{syn['synonym_name']}')
        for tmp_table in inspect(conn).get_temp_table_names():
            conn.exec_driver_sql(f'drop table {tmp_table}')