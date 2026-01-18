from django.core import checks
from django.db.backends.base.validation import BaseDatabaseValidation
from django.utils.version import get_docs_version
def _check_sql_mode(self, **kwargs):
    if not self.connection.sql_mode & {'STRICT_TRANS_TABLES', 'STRICT_ALL_TABLES'}:
        return [checks.Warning("%s Strict Mode is not set for database connection '%s'" % (self.connection.display_name, self.connection.alias), hint=("%s's Strict Mode fixes many data integrity problems in %s, such as data truncation upon insertion, by escalating warnings into errors. It is strongly recommended you activate it. See: https://docs.djangoproject.com/en/%s/ref/databases/#mysql-sql-mode" % (self.connection.display_name, self.connection.display_name, get_docs_version()),), id='mysql.W002')]
    return []