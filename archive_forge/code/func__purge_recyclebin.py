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
def _purge_recyclebin(eng, schema=None):
    with eng.begin() as conn:
        if schema is None:
            conn.exec_driver_sql('purge recyclebin')
        else:
            for owner, object_name, type_ in conn.exec_driver_sql("select owner, object_name,type from dba_recyclebin where owner=:schema and type='TABLE'", {'schema': conn.dialect.denormalize_name(schema)}).all():
                conn.exec_driver_sql(f'purge {type_} {owner}."{object_name}"')