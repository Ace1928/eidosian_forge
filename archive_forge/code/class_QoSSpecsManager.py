from cinderclient.apiclient import base as common_base
from cinderclient import base
class QoSSpecsManager(base.ManagerWithFind):
    """
    Manage :class:`QoSSpecs` resources.
    """
    resource_class = QoSSpecs

    def list(self, search_opts=None):
        """Get a list of all qos specs.

        :rtype: list of :class:`QoSSpecs`.
        """
        return self._list('/qos-specs', 'qos_specs')

    def get(self, qos_specs):
        """Get a specific qos specs.

        :param qos_specs: The ID of the :class:`QoSSpecs` to get.
        :rtype: :class:`QoSSpecs`
        """
        return self._get('/qos-specs/%s' % base.getid(qos_specs), 'qos_specs')

    def delete(self, qos_specs, force=False):
        """Delete a specific qos specs.

        :param qos_specs: The ID of the :class:`QoSSpecs` to be removed.
        :param force: Flag that indicates whether to delete target qos specs
                      if it was in-use.
        """
        return self._delete('/qos-specs/%s?force=%s' % (base.getid(qos_specs), force))

    def create(self, name, specs):
        """Create a qos specs.

        :param name: Descriptive name of the qos specs, must be unique
        :param specs: A dict of key/value pairs to be set
        :rtype: :class:`QoSSpecs`
        """
        body = {'qos_specs': {'name': name}}
        body['qos_specs'].update(specs)
        return self._create('/qos-specs', body, 'qos_specs')

    def set_keys(self, qos_specs, specs):
        """Add/Update keys in qos specs.

        :param qos_specs: The ID of qos specs
        :param specs: A dict of key/value pairs to be set
        :rtype: :class:`QoSSpecs`
        """
        body = {'qos_specs': {}}
        body['qos_specs'].update(specs)
        return self._update('/qos-specs/%s' % qos_specs, body)

    def unset_keys(self, qos_specs, specs):
        """Remove keys from a qos specs.

        :param qos_specs: The ID of qos specs
        :param specs: A list of key to be unset
        :rtype: :class:`QoSSpecs`
        """
        body = {'keys': specs}
        return self._update('/qos-specs/%s/delete_keys' % qos_specs, body)

    def get_associations(self, qos_specs):
        """Get associated entities of a qos specs.

        :param qos_specs: The id of the :class: `QoSSpecs`
        :return: a list of entities that associated with specific qos specs.
        """
        return self._list('/qos-specs/%s/associations' % base.getid(qos_specs), 'qos_associations')

    def associate(self, qos_specs, vol_type_id):
        """Associate a volume type with specific qos specs.

        :param qos_specs: The qos specs to be associated with
        :param vol_type_id: The volume type id to be associated with
        """
        resp, body = self.api.client.get('/qos-specs/%s/associate?vol_type_id=%s' % (base.getid(qos_specs), vol_type_id))
        return common_base.TupleWithMeta((resp, body), resp)

    def disassociate(self, qos_specs, vol_type_id):
        """Disassociate qos specs from volume type.

        :param qos_specs: The qos specs to be associated with
        :param vol_type_id: The volume type id to be associated with
        """
        resp, body = self.api.client.get('/qos-specs/%s/disassociate?vol_type_id=%s' % (base.getid(qos_specs), vol_type_id))
        return common_base.TupleWithMeta((resp, body), resp)

    def disassociate_all(self, qos_specs):
        """Disassociate all entities from specific qos specs.

        :param qos_specs: The qos specs to be associated with
        """
        resp, body = self.api.client.get('/qos-specs/%s/disassociate_all' % base.getid(qos_specs))
        return common_base.TupleWithMeta((resp, body), resp)