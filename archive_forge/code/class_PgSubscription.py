from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
class PgSubscription:
    """Class to work with PostgreSQL subscription.

    Args:
        module (AnsibleModule): Object of AnsibleModule class.
        cursor (cursor): Cursor object of psycopg library to work with PostgreSQL.
        name (str): The name of the subscription.
        db (str): The database name the subscription will be associated with.

    Attributes:
        module (AnsibleModule): Object of AnsibleModule class.
        cursor (cursor): Cursor object of psycopg library to work with PostgreSQL.
        name (str): Name of subscription.
        executed_queries (list): List of executed queries.
        attrs (dict): Dict with subscription attributes.
        exists (bool): Flag indicates the subscription exists or not.
    """

    def __init__(self, module, cursor, name, db):
        self.module = module
        self.cursor = cursor
        self.name = name
        self.db = db
        self.executed_queries = []
        self.attrs = {'owner': None, 'enabled': None, 'synccommit': None, 'conninfo': {}, 'slotname': None, 'publications': [], 'comment': None}
        self.empty_attrs = deepcopy(self.attrs)
        self.exists = self.check_subscr()

    def get_info(self):
        """Refresh the subscription information.

        Returns:
            ``self.attrs``.
        """
        self.exists = self.check_subscr()
        return self.attrs

    def check_subscr(self):
        """Check the subscription and refresh ``self.attrs`` subscription attribute.

        Returns:
            True if the subscription with ``self.name`` exists, False otherwise.
        """
        subscr_info = self.__get_general_subscr_info()
        if not subscr_info:
            self.attrs = deepcopy(self.empty_attrs)
            return False
        self.attrs['owner'] = subscr_info.get('rolname')
        self.attrs['enabled'] = subscr_info.get('subenabled')
        self.attrs['synccommit'] = subscr_info.get('subenabled')
        self.attrs['slotname'] = subscr_info.get('subslotname')
        self.attrs['publications'] = subscr_info.get('subpublications')
        if subscr_info.get('comment') is not None:
            self.attrs['comment'] = subscr_info.get('comment')
        else:
            self.attrs['comment'] = ''
        if subscr_info.get('subconninfo'):
            for param in subscr_info['subconninfo'].split(' '):
                tmp = param.split('=')
                try:
                    self.attrs['conninfo'][tmp[0]] = int(tmp[1])
                except ValueError:
                    self.attrs['conninfo'][tmp[0]] = tmp[1]
        return True

    def create(self, connparams, publications, subsparams, check_mode=True):
        """Create the subscription.

        Args:
            connparams (str): Connection string in libpq style.
            publications (list): Publications on the primary to use.
            subsparams (str): Parameters string in WITH () clause style.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            changed (bool): True if the subscription has been created, otherwise False.
        """
        query_fragments = []
        query_fragments.append("CREATE SUBSCRIPTION %s CONNECTION '%s' PUBLICATION %s" % (self.name, connparams, ', '.join(publications)))
        if subsparams:
            query_fragments.append('WITH (%s)' % subsparams)
        changed = self.__exec_sql(' '.join(query_fragments), check_mode=check_mode)
        return changed

    def update(self, connparams, publications, subsparams, check_mode=True):
        """Update the subscription.

        Args:
            connparams (dict): Connection dict in libpq style.
            publications (list): Publications on the primary to use.
            subsparams (dict): Dictionary of optional parameters.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            changed (bool): True if subscription has been updated, otherwise False.
        """
        changed = False
        if connparams:
            if connparams != self.attrs['conninfo']:
                changed = self.__set_conn_params(convert_conn_params(connparams), check_mode=check_mode)
        if publications:
            if sorted(self.attrs['publications']) != sorted(publications):
                changed = self.__set_publications(publications, check_mode=check_mode)
        if subsparams:
            params_to_update = []
            for param, value in iteritems(subsparams):
                if param == 'enabled':
                    if self.attrs['enabled'] and value is False:
                        changed = self.enable(enabled=False, check_mode=check_mode)
                    elif not self.attrs['enabled'] and value is True:
                        changed = self.enable(enabled=True, check_mode=check_mode)
                elif param == 'synchronous_commit':
                    if self.attrs['synccommit'] is True and value is False:
                        params_to_update.append('%s = false' % param)
                    elif self.attrs['synccommit'] is False and value is True:
                        params_to_update.append('%s = true' % param)
                elif param == 'slot_name':
                    if self.attrs['slotname'] and self.attrs['slotname'] != value:
                        params_to_update.append('%s = %s' % (param, value))
                else:
                    self.module.warn("Parameter '%s' is not in params supported for update '%s', ignored..." % (param, SUBSPARAMS_KEYS_FOR_UPDATE))
            if params_to_update:
                changed = self.__set_params(params_to_update, check_mode=check_mode)
        return changed

    def drop(self, cascade=False, check_mode=True):
        """Drop the subscription.

        Kwargs:
            cascade (bool): Flag indicates that the subscription needs to be deleted
                with its dependencies.
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            changed (bool): True if the subscription has been removed, otherwise False.
        """
        if self.exists:
            query_fragments = ['DROP SUBSCRIPTION %s' % self.name]
            if cascade:
                query_fragments.append('CASCADE')
            return self.__exec_sql(' '.join(query_fragments), check_mode=check_mode)

    def set_owner(self, role, check_mode=True):
        """Set a subscription owner.

        Args:
            role (str): Role (user) name that needs to be set as a subscription owner.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
        query = 'ALTER SUBSCRIPTION %s OWNER TO "%s"' % (self.name, role)
        return self.__exec_sql(query, check_mode=check_mode)

    def set_comment(self, comment, check_mode=True):
        """Set a subscription comment.

        Args:
            comment (str): Comment to set on the subscription.

        Kwargs:
            check_mode (bool): If True, don not change anything.

        Returns:
            True if success, False otherwise.
        """
        set_comment(self.cursor, comment, 'subscription', self.name, check_mode, self.executed_queries)
        return True

    def refresh(self, check_mode=True):
        """Refresh publication.

        Fetches missing table info from publisher.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
        query = 'ALTER SUBSCRIPTION %s REFRESH PUBLICATION' % self.name
        return self.__exec_sql(query, check_mode=check_mode)

    def __set_params(self, params_to_update, check_mode=True):
        """Update optional subscription parameters.

        Args:
            params_to_update (list): Parameters with values to update.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
        query = 'ALTER SUBSCRIPTION %s SET (%s)' % (self.name, ', '.join(params_to_update))
        return self.__exec_sql(query, check_mode=check_mode)

    def __set_conn_params(self, connparams, check_mode=True):
        """Update connection parameters.

        Args:
            connparams (str): Connection string in libpq style.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
        query = "ALTER SUBSCRIPTION %s CONNECTION '%s'" % (self.name, connparams)
        return self.__exec_sql(query, check_mode=check_mode)

    def __set_publications(self, publications, check_mode=True):
        """Update publications.

        Args:
            publications (list): Publications on the primary to use.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
        query = 'ALTER SUBSCRIPTION %s SET PUBLICATION %s' % (self.name, ', '.join(publications))
        return self.__exec_sql(query, check_mode=check_mode)

    def enable(self, enabled=True, check_mode=True):
        """Enable or disable the subscription.

        Kwargs:
            enable (bool): Flag indicates that the subscription needs
                to be enabled or disabled.
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
        if enabled:
            query = 'ALTER SUBSCRIPTION %s ENABLE' % self.name
        else:
            query = 'ALTER SUBSCRIPTION %s DISABLE' % self.name
        return self.__exec_sql(query, check_mode=check_mode)

    def __get_general_subscr_info(self):
        """Get and return general subscription information.

        Returns:
            Dict with subscription information if successful, False otherwise.
        """
        query = "SELECT obj_description(s.oid, 'pg_subscription') AS comment, d.datname, r.rolname, s.subenabled, s.subconninfo, s.subslotname, s.subsynccommit, s.subpublications FROM pg_catalog.pg_subscription s JOIN pg_catalog.pg_database d ON s.subdbid = d.oid JOIN pg_catalog.pg_roles AS r ON s.subowner = r.oid WHERE s.subname = %(name)s AND d.datname = %(db)s"
        result = exec_sql(self, query, query_params={'name': self.name, 'db': self.db}, add_to_executed=False)
        if result:
            return result[0]
        else:
            return False

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