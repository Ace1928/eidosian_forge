from ... import exc
from ...testing.provision import configure_follower
from ...testing.provision import create_db
from ...testing.provision import drop_db
from ...testing.provision import generate_driver_url
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import upsert
@configure_follower.for_db('mysql', 'mariadb')
def _mysql_configure_follower(config, ident):
    config.test_schema = '%s_test_schema' % ident
    config.test_schema_2 = '%s_test_schema_2' % ident