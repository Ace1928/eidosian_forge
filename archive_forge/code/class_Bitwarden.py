from __future__ import (absolute_import, division, print_function)
from subprocess import Popen, PIPE
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.parsing.ajson import AnsibleJSONDecoder
from ansible.plugins.lookup import LookupBase
class Bitwarden(object):

    def __init__(self, path='bw'):
        self._cli_path = path
        self._session = None

    @property
    def cli_path(self):
        return self._cli_path

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, value):
        self._session = value

    @property
    def unlocked(self):
        out, err = self._run(['status'], stdin='')
        decoded = AnsibleJSONDecoder().raw_decode(out)[0]
        return decoded['status'] == 'unlocked'

    def _run(self, args, stdin=None, expected_rc=0):
        if self.session:
            args += ['--session', self.session]
        p = Popen([self.cli_path] + args, stdout=PIPE, stderr=PIPE, stdin=PIPE)
        out, err = p.communicate(to_bytes(stdin))
        rc = p.wait()
        if rc != expected_rc:
            if len(args) > 2 and args[0] == 'get' and (args[1] == 'item') and (b'Not found.' in err):
                return ('null', '')
            raise BitwardenException(err)
        return (to_text(out, errors='surrogate_or_strict'), to_text(err, errors='surrogate_or_strict'))

    def _get_matches(self, search_value, search_field, collection_id=None):
        """Return matching records whose search_field is equal to key.
        """
        if search_value:
            if search_field == 'id':
                params = ['get', 'item', search_value]
            else:
                params = ['list', 'items', '--search', search_value]
            if collection_id:
                params.extend(['--collectionid', collection_id])
        else:
            if not collection_id:
                raise AnsibleError('search_value is required if collection_id is not set.')
            params = ['list', 'items', '--collectionid', collection_id]
        out, err = self._run(params)
        initial_matches = AnsibleJSONDecoder().raw_decode(out)[0]
        if search_field == 'id' or not search_value:
            if initial_matches is None:
                initial_matches = []
            else:
                initial_matches = [initial_matches]
        return [item for item in initial_matches if item[search_field] == search_value]

    def get_field(self, field, search_value=None, search_field='name', collection_id=None):
        """Return a list of the specified field for records whose search_field match search_value
        and filtered by collection if collection has been provided.

        If field is None, return the whole record for each match.
        """
        matches = self._get_matches(search_value, search_field, collection_id)
        if not field:
            return matches
        field_matches = []
        for match in matches:
            if 'fields' in match:
                custom_field_found = False
                for custom_field in match['fields']:
                    if field == custom_field['name']:
                        field_matches.append(custom_field['value'])
                        custom_field_found = True
                        break
                if custom_field_found:
                    continue
            if 'login' in match and field in match['login']:
                field_matches.append(match['login'][field])
                continue
            if field in match:
                field_matches.append(match[field])
                continue
        if matches and (not field_matches):
            raise AnsibleError('field {field} does not exist in {search_value}'.format(field=field, search_value=search_value))
        return field_matches