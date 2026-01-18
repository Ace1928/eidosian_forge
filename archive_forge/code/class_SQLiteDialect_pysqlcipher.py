from .pysqlite import SQLiteDialect_pysqlite
from ... import pool
class SQLiteDialect_pysqlcipher(SQLiteDialect_pysqlite):
    driver = 'pysqlcipher'
    supports_statement_cache = True
    pragmas = ('kdf_iter', 'cipher', 'cipher_page_size', 'cipher_use_hmac')

    @classmethod
    def import_dbapi(cls):
        try:
            import sqlcipher3 as sqlcipher
        except ImportError:
            pass
        else:
            return sqlcipher
        from pysqlcipher3 import dbapi2 as sqlcipher
        return sqlcipher

    @classmethod
    def get_pool_class(cls, url):
        return pool.SingletonThreadPool

    def on_connect_url(self, url):
        super_on_connect = super().on_connect_url(url)
        passphrase = url.password or ''
        url_query = dict(url.query)

        def on_connect(conn):
            cursor = conn.cursor()
            cursor.execute('pragma key="%s"' % passphrase)
            for prag in self.pragmas:
                value = url_query.get(prag, None)
                if value is not None:
                    cursor.execute('pragma %s="%s"' % (prag, value))
            cursor.close()
            if super_on_connect:
                super_on_connect(conn)
        return on_connect

    def create_connect_args(self, url):
        plain_url = url._replace(password=None)
        plain_url = plain_url.difference_update_query(self.pragmas)
        return super().create_connect_args(plain_url)