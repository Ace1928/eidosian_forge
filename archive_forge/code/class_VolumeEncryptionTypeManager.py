from cinderclient.apiclient import base as common_base
from cinderclient import base
class VolumeEncryptionTypeManager(base.ManagerWithFind):
    """
    Manage :class: `VolumeEncryptionType` resources.
    """
    resource_class = VolumeEncryptionType

    def list(self, search_opts=None):
        """
        List all volume encryption types.

        :param search_opts: Search options to filter out volume
                            encryption types
        :return: a list of :class: VolumeEncryptionType instances
        """
        volume_types = self.api.volume_types.list()
        encryption_types = []
        list_of_resp = []
        for volume_type in volume_types:
            encryption_type = self._get('/types/%s/encryption' % base.getid(volume_type))
            if hasattr(encryption_type, 'volume_type_id'):
                encryption_types.append(encryption_type)
            list_of_resp.extend(encryption_type.request_ids)
        return common_base.ListWithMeta(encryption_types, list_of_resp)

    def get(self, volume_type):
        """
        Get the volume encryption type for the specified volume type.

        :param volume_type: the volume type to query
        :return: an instance of :class: VolumeEncryptionType
        """
        return self._get('/types/%s/encryption' % base.getid(volume_type))

    def create(self, volume_type, specs):
        """
        Creates encryption type for a volume type. Default: admin only.

        :param volume_type: the volume type on which to add an encryption type
        :param specs: the encryption type specifications to add
        :return: an instance of :class: VolumeEncryptionType
        """
        body = {'encryption': specs}
        return self._create('/types/%s/encryption' % base.getid(volume_type), body, 'encryption')

    def update(self, volume_type, specs):
        """
        Update the encryption type information for the specified volume type.

        :param volume_type: the volume type whose encryption type information
                            must be updated
        :param specs: the encryption type specifications to update
        :return: an instance of :class: VolumeEncryptionType
        """
        body = {'encryption': specs}
        return self._update('/types/%s/encryption/provider' % base.getid(volume_type), body)

    def delete(self, volume_type):
        """
        Delete the encryption type information for the specified volume type.

        :param volume_type: the volume type whose encryption type information
                            must be deleted
        """
        return self._delete('/types/%s/encryption/provider' % base.getid(volume_type))