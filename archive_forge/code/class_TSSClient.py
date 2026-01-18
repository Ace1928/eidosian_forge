from __future__ import absolute_import, division, print_function
import abc
import os
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils import six
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
@six.add_metaclass(abc.ABCMeta)
class TSSClient(object):

    def __init__(self):
        self._client = None

    @staticmethod
    def from_params(**server_parameters):
        if HAS_TSS_AUTHORIZER:
            return TSSClientV1(**server_parameters)
        else:
            return TSSClientV0(**server_parameters)

    def get_secret(self, term, secret_path, fetch_file_attachments, file_download_path):
        display.debug('tss_lookup term: %s' % term)
        secret_id = self._term_to_secret_id(term)
        if secret_id == 0 and secret_path:
            fetch_secret_by_path = True
            display.vvv(u'Secret Server lookup of Secret with path %s' % secret_path)
        else:
            fetch_secret_by_path = False
            display.vvv(u'Secret Server lookup of Secret with ID %d' % secret_id)
        if fetch_file_attachments:
            if fetch_secret_by_path:
                obj = self._client.get_secret_by_path(secret_path, fetch_file_attachments)
            else:
                obj = self._client.get_secret(secret_id, fetch_file_attachments)
            for i in obj['items']:
                if file_download_path and os.path.isdir(file_download_path):
                    if i['isFile']:
                        try:
                            file_content = i['itemValue'].content
                            with open(os.path.join(file_download_path, str(obj['id']) + '_' + i['slug']), 'wb') as f:
                                f.write(file_content)
                        except ValueError:
                            raise AnsibleOptionsError('Failed to download {0}'.format(str(i['slug'])))
                        except AttributeError:
                            display.warning('Could not read file content for {0}'.format(str(i['slug'])))
                        finally:
                            i['itemValue'] = '*** Not Valid For Display ***'
                else:
                    raise AnsibleOptionsError('File download path does not exist')
            return obj
        elif fetch_secret_by_path:
            return self._client.get_secret_by_path(secret_path, False)
        else:
            return self._client.get_secret_json(secret_id)

    def get_secret_ids_by_folderid(self, term):
        display.debug('tss_lookup term: %s' % term)
        folder_id = self._term_to_folder_id(term)
        display.vvv(u"Secret Server lookup of Secret id's with Folder ID %d" % folder_id)
        return self._client.get_secret_ids_by_folderid(folder_id)

    @staticmethod
    def _term_to_secret_id(term):
        try:
            return int(term)
        except ValueError:
            raise AnsibleOptionsError('Secret ID must be an integer')

    @staticmethod
    def _term_to_folder_id(term):
        try:
            return int(term)
        except ValueError:
            raise AnsibleOptionsError('Folder ID must be an integer')