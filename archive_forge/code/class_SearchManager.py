import csv
import difflib
from io import StringIO
from lxml import etree
from .jsonutil import JsonTable, get_column, get_where, get_selection
from .errors import is_xnat_error, catch_error
from .errors import ProgrammingError, NotSupportedError
from .errors import DataError, DatabaseError
class SearchManager(object):
    """ Search interface.
        Handles operations to save and get back searches on the server.

        Examples
        --------
            >>> row = 'xnat:subjectData'
            >>> columns = ['xnat:subjectData/PROJECT',
                           'xnat:subjectData/SUBJECT_ID'
                           ]
            >>> criteria = [('xnat:subjectData/SUBJECT_ID', 'LIKE', '*'),
                            'AND'
                            ]
            >>> interface.manage.search.save('mysearch', row, columns,
                                             criteria, sharing='public',
                                             description='my first search'
                                             )
    """

    def __init__(self, interface):
        self._intf = interface

    def _save_search(self, row, columns, constraints, name, desc, sharing):
        self._intf._get_entry_point()
        name = name.replace(' ', '_')
        if sharing == 'private':
            users = [self._intf._user]
        elif sharing == 'public':
            users = []
        elif isinstance(sharing, list):
            users = sharing
        else:
            raise NotSupportedError('Share mode %s not valid' % sharing)
        self._intf._exec('%s/search/saved/%s?inbody=true' % (self._intf._entry, name), method='PUT', body=build_search_document(row, columns, constraints, name, desc.replace('%', '%%'), users))

    def save(self, name, row, columns, constraints, sharing='private', description=''):
        """ Saves a query on the XNAT server.

            Parameters
            ----------
            name: string
                Name of the query displayed on the Web Interface and
                used to get back the results.
            row: string
                Datatype from `Interface.inspect.datatypes()`.
                Usually ``xnat:subjectData``
            columns: list
                List of data fields from
                `Interface.inspect.datatypes('*', '*')`
            constraints: list
                See also: `Search.where()`
            sharing: string | list
                Define by whom the query is visible.
                If sharing is a string it may be either
                ``private`` or ``public``.
                Otherwise a list of valid logins for the XNAT server
                from `Interface.users()`.

            See Also
            --------
            :func:`Search.where`
        """
        self._save_search(row, columns, constraints, name, description, sharing)

    def saved(self, with_description=False):
        """ Returns the names of accessible saved search on the server.
        """
        self._intf._get_entry_point()
        jdata = self._intf._get_json('%s/search/saved?format=json' % self._intf._entry)
        if with_description:
            return [(ld['brief_description'], ld['description'].replace('%%', '%')) for ld in get_selection(jdata, ['brief_description', 'description']) if not ld['brief_description'].startswith('template_')]
        else:
            return [name for name in get_column(jdata, 'brief_description') if not name.startswith('template_')]

    def get(self, name, out_format='results'):
        """ Returns the results of the query saved on the XNAT server or
            the query itself to know what it does.

            Parameters
            ----------
            name: string
                Name of the saved search. An exception is raised if the name
                does not exist.
            out_format: string
                Can take the following values:
                    - results to download the results of the search
                    - xml to download the XML document defining the search
                    - query to get the pyxnat representation of the search
        """
        self._intf._get_entry_point()
        jdata = self._intf._get_json('%s/search/saved?format=json' % self._intf._entry)
        try:
            search_id = get_where(jdata, brief_description=name)[0]['id']
        except IndexError:
            raise DatabaseError('%s not found' % name)
        if out_format in ['xml', 'query']:
            bundle = self._intf._exec('%s/search/saved/%s' % (self._intf._entry, search_id), 'GET')
            if out_format == 'xml':
                return bundle
            else:
                return query_from_xml(bundle)
        content = self._intf._exec('%s/search/saved/%s/results?format=csv' % (self._intf._entry, search_id), 'GET')
        results = csv.reader(StringIO(content.decode('utf-8')), delimiter=',', quotechar='"')
        headers = next(results)
        return JsonTable([dict(zip(headers, res)) for res in results], headers)

    def delete(self, name):
        """ Removes the search from the server.
        """
        self._intf._get_entry_point()
        jdata = self._intf._get_json('%s/search/saved?format=json' % self._intf._entry)
        try:
            search_id = get_where(jdata, brief_description=name)[0]['id']
        except IndexError:
            raise DatabaseError('%s not found' % name)
        self._intf._exec('%s/search/saved/%s' % (self._intf._entry, search_id), 'DELETE')

    def save_template(self, name, row=None, columns=[], constraints=[], sharing='private', description=''):
        """
            Define and save a search template. Same as the save method, but
            the values in the constraints are used as keywords for value
            replacement when using the template.

            Parameters
            ----------
            name: string
                Name under which the template is save in XNAT. A template is
                prepended to the name so that it appear clearly as a template
                on the web interface.
            row: string
                Datatype from `Interface.inspect.datatypes()`.
                Usually ``xnat:subjectData``
            columns: list
                List of data fields from
                `Interface.inspect.datatypes('*', '*')`
            constraints: list
                See also: `Search.where()`, values are keywords for the
                template
            sharing: string | list
                Define by whom the query is visible.
                If sharing is a string it may be either
                ``private`` or ``public``.
                Otherwise a list of valid logins for the XNAT server
                from `Interface.users()`.
        """

        def _make_template(query):
            query_template = []
            for constraint in query:
                if isinstance(constraint, tuple):
                    query_template.append((constraint[0], constraint[1], '%%(%s)s' % constraint[2]))
                elif isinstance(constraint, str):
                    query_template.append(constraint)
                elif isinstance(constraint, list):
                    query_template.append(_make_template(constraint))
                else:
                    raise ProgrammingError('Unrecognized token in query: %s' % constraint)
            return query_template
        self._save_search(row, columns, _make_template(constraints), 'template_%s' % name, description, sharing)

    def saved_templates(self, with_description=False):
        """ Returns the names of accessible saved search templates on the server.
        """
        self._intf._get_entry_point()
        jdata = self._intf._get_json('%s/search/saved?format=json' % self._intf._entry)
        if with_description:
            return [(ld['brief_description'].split('template_')[1], ld['description'].replace('%%', '%')) for ld in get_selection(jdata, ['brief_description', 'description']) if ld['brief_description'].startswith('template_')]
        else:
            return [name.split('template_')[1] for name in get_column(jdata, 'brief_description') if name.startswith('template_')]

    def use_template(self, name, values):
        """
            Performs a search query using a previously saved template.

            Parameters
            ----------
            name: string
                Name of the template.
            values: dict
                Values to put in the template, get the valid keys using
                the get_template method.

            Examples
            --------
            >>> interface.manage.search.use_template(name,
                          {'subject_id':'ID',
                           'age':'32'
                           })

        """
        self._intf._get_entry_point()
        bundle = self.get_template(name, True) % values
        _query = query_from_xml(bundle)
        bundle = build_search_document(_query['row'], _query['columns'], _query['constraints'])
        content = self._intf._exec('%s/search?format=csv' % self._intf._entry, 'POST', bundle)
        results = csv.reader(StringIO(content), delimiter=',', quotechar='"')
        headers = results.next()
        return JsonTable([dict(zip(headers, res)) for res in results], headers)

    def get_template(self, name, as_xml=False):
        """ Get a saved template, either as an xml document, or as a pyxnat
            representation, with the keys to be used in the template
            between the parentheses in %()s.

            Parameters
            ----------
            name: str
                Name under which the template is saved
            as_xml: boolean
                If True returns an XML document, else return a list of
                constraints. Defaults to False.
        """
        self._intf._get_entry_point()
        jdata = self._intf._get_json('%s/search/saved?format=json' % self._intf._entry)
        try:
            search_id = get_where(jdata, brief_description='template_%s' % name)[0]['id']
        except IndexError:
            raise DatabaseError('%s not found' % name)
        bundle = self._intf._exec('%s/search/saved/%s' % (self._intf._entry, search_id), 'GET')
        if as_xml:
            return bundle
        else:
            _query = query_from_xml(bundle)
            return (_query['row'], _query['columns'], _query['constraints'], _query['description'])

    def delete_template(self, name):
        """ Deletes a search template.
        """
        self.delete('template_%s' % name)

    def eval_rpn_exp(self, rpnexp):
        return rpn_contraints(rpnexp)