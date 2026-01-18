from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
class PgPublication:
    """Class to work with PostgreSQL publication.

    Args:
        module (AnsibleModule): Object of AnsibleModule class.
        cursor (cursor): Cursor object of psycopg library to work with PostgreSQL.
        name (str): The name of the publication.

    Attributes:
        module (AnsibleModule): Object of AnsibleModule class.
        cursor (cursor): Cursor object of psycopg library to work with PostgreSQL.
        name (str): Name of the publication.
        executed_queries (list): List of executed queries.
        attrs (dict): Dict with publication attributes.
        exists (bool): Flag indicates the publication exists or not.
    """

    def __init__(self, module, cursor, name):
        self.module = module
        self.cursor = cursor
        self.name = name
        self.executed_queries = []
        self.attrs = {'alltables': False, 'tables': [], 'parameters': {}, 'owner': ''}
        self.exists = self.check_pub()

    def get_info(self):
        """Refresh the publication information.

        Returns:
            ``self.attrs``.
        """
        self.exists = self.check_pub()
        return self.attrs

    def check_pub(self):
        """Check the publication and refresh ``self.attrs`` publication attribute.

        Returns:
            True if the publication with ``self.name`` exists, False otherwise.
        """
        pub_info = self.__get_general_pub_info()
        if not pub_info:
            return False
        self.attrs['owner'] = pub_info.get('pubowner')
        self.attrs['comment'] = pub_info.get('comment') if pub_info.get('comment') is not None else ''
        self.attrs['parameters']['publish'] = {}
        self.attrs['parameters']['publish']['insert'] = pub_info.get('pubinsert', False)
        self.attrs['parameters']['publish']['update'] = pub_info.get('pubupdate', False)
        self.attrs['parameters']['publish']['delete'] = pub_info.get('pubdelete', False)
        if pub_info.get('pubtruncate'):
            self.attrs['parameters']['publish']['truncate'] = pub_info.get('pubtruncate')
        if not pub_info.get('puballtables'):
            table_info = self.__get_tables_pub_info()
            for i, schema_and_table in enumerate(table_info):
                table_info[i] = pg_quote_identifier(schema_and_table['schema_dot_table'], 'table')
            self.attrs['tables'] = table_info
        else:
            self.attrs['alltables'] = True
        return True

    def create(self, tables, params, owner, comment, check_mode=True):
        """Create the publication.

        Args:
            tables (list): List with names of the tables that need to be added to the publication.
            params (dict): Dict contains optional publication parameters and their values.
            owner (str): Name of the publication owner.
            comment (str): Comment on the publication.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            changed (bool): True if publication has been created, otherwise False.
        """
        changed = True
        query_fragments = ['CREATE PUBLICATION %s' % pg_quote_identifier(self.name, 'publication')]
        if tables:
            query_fragments.append('FOR TABLE %s' % ', '.join(tables))
        else:
            query_fragments.append('FOR ALL TABLES')
        if params:
            params_list = []
            for key, val in iteritems(params):
                params_list.append("%s = '%s'" % (key, val))
            query_fragments.append('WITH (%s)' % ', '.join(params_list))
        changed = self.__exec_sql(' '.join(query_fragments), check_mode=check_mode)
        if owner:
            self.__pub_set_owner(owner, check_mode=check_mode)
        if comment is not None:
            set_comment(self.cursor, comment, 'publication', self.name, check_mode, self.executed_queries)
        return changed

    def update(self, tables, params, owner, comment, check_mode=True):
        """Update the publication.

        Args:
            tables (list): List with names of the tables that need to be presented in the publication.
            params (dict): Dict contains optional publication parameters and their values.
            owner (str): Name of the publication owner.
            comment (str): Comment on the publication.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            changed (bool): True if publication has been updated, otherwise False.
        """
        changed = False
        if tables and (not self.attrs['alltables']):
            for tbl in tables:
                if tbl not in self.attrs['tables']:
                    changed = self.__pub_add_table(tbl, check_mode=check_mode)
            for tbl in self.attrs['tables']:
                if tbl not in tables:
                    changed = self.__pub_drop_table(tbl, check_mode=check_mode)
        elif tables and self.attrs['alltables']:
            changed = self.__pub_set_tables(tables, check_mode=check_mode)
        if params:
            for key, val in iteritems(params):
                if self.attrs['parameters'].get(key):
                    if key == 'publish':
                        val_dict = self.attrs['parameters']['publish'].copy()
                        val_list = val.split(',')
                        for v in val_dict:
                            if v in val_list:
                                val_dict[v] = True
                            else:
                                val_dict[v] = False
                        if val_dict != self.attrs['parameters']['publish']:
                            changed = self.__pub_set_param(key, val, check_mode=check_mode)
                    elif self.attrs['parameters'][key] != val:
                        changed = self.__pub_set_param(key, val, check_mode=check_mode)
                else:
                    changed = self.__pub_set_param(key, val, check_mode=check_mode)
        if owner:
            if owner != self.attrs['owner']:
                changed = self.__pub_set_owner(owner, check_mode=check_mode)
        if comment is not None and comment != self.attrs['comment']:
            changed = set_comment(self.cursor, comment, 'publication', self.name, check_mode, self.executed_queries)
        return changed

    def drop(self, cascade=False, check_mode=True):
        """Drop the publication.

        Kwargs:
            cascade (bool): Flag indicates that publication needs to be deleted
                with its dependencies.
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            changed (bool): True if publication has been updated, otherwise False.
        """
        if self.exists:
            query_fragments = []
            query_fragments.append('DROP PUBLICATION %s' % pg_quote_identifier(self.name, 'publication'))
            if cascade:
                query_fragments.append('CASCADE')
            return self.__exec_sql(' '.join(query_fragments), check_mode=check_mode)

    def __get_general_pub_info(self):
        """Get and return general publication information.

        Returns:
            Dict with publication information if successful, False otherwise.
        """
        pgtrunc_sup = exec_sql(self, "SELECT 1 FROM information_schema.columns WHERE table_name = 'pg_publication' AND column_name = 'pubtruncate'", add_to_executed=False)
        if pgtrunc_sup:
            query = "SELECT obj_description(p.oid, 'pg_publication') AS comment, r.rolname AS pubowner, p.puballtables, p.pubinsert, p.pubupdate , p.pubdelete, p.pubtruncate FROM pg_publication AS p JOIN pg_catalog.pg_roles AS r ON p.pubowner = r.oid WHERE p.pubname = %(pname)s"
        else:
            query = "SELECT obj_description(p.oid, 'pg_publication') AS comment, r.rolname AS pubowner, p.puballtables, p.pubinsert, p.pubupdate , p.pubdelete FROM pg_publication AS p JOIN pg_catalog.pg_roles AS r ON p.pubowner = r.oid WHERE p.pubname = %(pname)s"
        result = exec_sql(self, query, query_params={'pname': self.name}, add_to_executed=False)
        if result:
            return result[0]
        else:
            return False

    def __get_tables_pub_info(self):
        """Get and return tables that are published by the publication.

        Returns:
            List of dicts with published tables.
        """
        query = "SELECT schemaname || '.' || tablename as schema_dot_table FROM pg_publication_tables WHERE pubname = %(pname)s"
        return exec_sql(self, query, query_params={'pname': self.name}, add_to_executed=False)

    def __pub_add_table(self, table, check_mode=False):
        """Add a table to the publication.

        Args:
            table (str): Table name.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
        query = 'ALTER PUBLICATION %s ADD TABLE %s' % (pg_quote_identifier(self.name, 'publication'), pg_quote_identifier(table, 'table'))
        return self.__exec_sql(query, check_mode=check_mode)

    def __pub_drop_table(self, table, check_mode=False):
        """Drop a table from the publication.

        Args:
            table (str): Table name.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
        query = 'ALTER PUBLICATION %s DROP TABLE %s' % (pg_quote_identifier(self.name, 'publication'), pg_quote_identifier(table, 'table'))
        return self.__exec_sql(query, check_mode=check_mode)

    def __pub_set_tables(self, tables, check_mode=False):
        """Set a table suit that need to be published by the publication.

        Args:
            tables (list): List of tables.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
        quoted_tables = [pg_quote_identifier(t, 'table') for t in tables]
        query = 'ALTER PUBLICATION %s SET TABLE %s' % (pg_quote_identifier(self.name, 'publication'), ', '.join(quoted_tables))
        return self.__exec_sql(query, check_mode=check_mode)

    def __pub_set_param(self, param, value, check_mode=False):
        """Set an optional publication parameter.

        Args:
            param (str): Name of the parameter.
            value (str): Parameter value.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
        query = "ALTER PUBLICATION %s SET (%s = '%s')" % (pg_quote_identifier(self.name, 'publication'), param, value)
        return self.__exec_sql(query, check_mode=check_mode)

    def __pub_set_owner(self, role, check_mode=False):
        """Set a publication owner.

        Args:
            role (str): Role (user) name that needs to be set as a publication owner.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
        query = 'ALTER PUBLICATION %s OWNER TO "%s"' % (pg_quote_identifier(self.name, 'publication'), role)
        return self.__exec_sql(query, check_mode=check_mode)

    def __exec_sql(self, query, check_mode=False):
        """Execute SQL query.

        Note: If we need just to get information from the database,
            we use ``exec_sql`` function directly.

        Args:
            query (str): Query that needs to be executed.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just add ``query`` to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
        if check_mode:
            self.executed_queries.append(query)
            return True
        else:
            return exec_sql(self, query, return_bool=True)