from ... import exc
from ...testing.provision import configure_follower
from ...testing.provision import create_db
from ...testing.provision import drop_db
from ...testing.provision import generate_driver_url
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import upsert
@temp_table_keyword_args.for_db('mysql', 'mariadb')
def _mysql_temp_table_keyword_args(cfg, eng):
    return {'prefixes': ['TEMPORARY']}