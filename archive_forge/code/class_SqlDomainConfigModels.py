from keystone.common import sql
from keystone.resource.config_backends import sql as config_sql
from keystone.tests import unit
from keystone.tests.unit.backend import core_sql
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.resource import test_core
class SqlDomainConfigModels(core_sql.BaseBackendSqlModels):

    def test_whitelisted_model(self):
        cols = (('domain_id', sql.String, 64), ('group', sql.String, 255), ('option', sql.String, 255), ('value', sql.JsonBlob, None))
        self.assertExpectedSchema('whitelisted_config', cols)

    def test_sensitive_model(self):
        cols = (('domain_id', sql.String, 64), ('group', sql.String, 255), ('option', sql.String, 255), ('value', sql.JsonBlob, None))
        self.assertExpectedSchema('sensitive_config', cols)