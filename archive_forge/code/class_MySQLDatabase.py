import sys
import numpy as np
from pymysql import connect
from pymysql.err import ProgrammingError
from copy import deepcopy
from ase.db.sqlite import SQLite3Database
from ase.db.sqlite import init_statements
from ase.db.sqlite import VERSION
from ase.db.postgresql import remove_nan_and_inf, insert_nan_and_inf
import ase.io.jsonio
import json
class MySQLDatabase(SQLite3Database):
    """
    ASE interface to a MySQL database (via pymysql package).

    Arguments
    ==========
    url: str
        URL to the database. It should have the form
        mysql://username:password@host:port/database_name.
        Example URL with the following credentials
            username: john
            password: johnspasswd
            host: localhost (i.e. server is running locally)
            database: johns_calculations
            port: 3306
        mysql://john:johnspasswd@localhost:3306/johns_calculations
    create_indices: bool
        Carried over from parent class. Currently indices are not
        created for MySQL, as TEXT fields cannot be hashed by MySQL.
    use_lock_file: bool
        See SQLite
    serial: bool
        See SQLite
    """
    type = 'mysql'
    default = 'DEFAULT'

    def __init__(self, url=None, create_indices=True, use_lock_file=False, serial=False):
        super(MySQLDatabase, self).__init__(url, create_indices, use_lock_file, serial)
        self.host = None
        self.username = None
        self.passwd = None
        self.db_name = None
        self.port = 3306
        self._parse_url(url)

    def _parse_url(self, url):
        """
        Parse the URL
        """
        url = url.replace('mysql://', '')
        url = url.replace('mariadb://', '')
        splitted = url.split(':', 1)
        self.username = splitted[0]
        splitted = splitted[1].split('@')
        self.passwd = splitted[0]
        splitted = splitted[1].split('/')
        host_and_port = splitted[0].split(':')
        self.host = host_and_port[0]
        self.port = int(host_and_port[1])
        self.db_name = splitted[1]

    def _connect(self):
        return Connection(host=self.host, user=self.username, passwd=self.passwd, db_name=self.db_name, port=self.port, binary_prefix=True)

    def _initialize(self, con):
        if self.initialized:
            return
        cur = con.cursor()
        information_exists = True
        self._metadata = {}
        try:
            cur.execute('SELECT 1 FROM information')
        except ProgrammingError:
            information_exists = False
        if not information_exists:
            init_statements_cpy = deepcopy(init_statements)
            init_statements_cpy[0] = init_statements_cpy[0][:-1] + ', PRIMARY KEY(id))'
            statements = schema_update(init_statements_cpy)
            for statement in statements:
                cur.execute(statement)
            con.commit()
            self.version = VERSION
        else:
            cur.execute('select * from information')
            for name, value in cur.fetchall():
                if name == 'version':
                    self.version = int(value)
                elif name == 'metadata':
                    self._metadata = json.loads(value)
        self.initialized = True

    def blob(self, array):
        if array is None:
            return None
        return super(MySQLDatabase, self).blob(array).tobytes()

    def get_offset_string(self, offset, limit=None):
        sql = ''
        if not limit:
            sql += '\nLIMIT 10000000000'
        sql += '\nOFFSET {0}'.format(offset)
        return sql

    def get_last_id(self, cur):
        cur.execute('select max(id) as ID from systems')
        last_id = cur.fetchone()[0]
        return last_id

    def create_select_statement(self, keys, cmps, sort=None, order=None, sort_table=None, what='systems.*'):
        sql, value = super(MySQLDatabase, self).create_select_statement(keys, cmps, sort, order, sort_table, what)
        for subst in MySQLCursor.sql_replace:
            sql = sql.replace(subst[0], subst[1])
        return (sql, value)

    def encode(self, obj, binary=False):
        return ase.io.jsonio.encode(remove_nan_and_inf(obj))

    def decode(self, obj, lazy=False):
        return insert_nan_and_inf(ase.io.jsonio.decode(obj))