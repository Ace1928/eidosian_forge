import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class _PerOperationACL(ACLFormatter):

    def __init__(self, parent_acl, entity_ref=None, users=None, project_access=None, operation_type=None, created=None, updated=None):
        """Per Operation ACL data instance for secret or container.

        This class not to be instantiated outside of this module.

        :param parent_acl: acl entity to this per operation data belongs to
        :param str entity_ref: Full HATEOAS reference to a secret or container
        :param users: List of Keystone userid(s) to be used for ACL.
        :type users: List or None
        :param bool project_access: Flag indicating project access behavior
        :param str operation_type: Type indicating which class of Barbican
            operations this ACL is defined for e.g. 'read' operations
        :param str created: Time string indicating ACL create timestamp. This
            is populated only when populating data from api response. Not
            needed in client input.
        :param str updated: Time string indicating ACL last update timestamp.
            This is populated only when populating data from api response. Not
            needed in client input.
        """
        self._parent_acl = parent_acl
        self._entity_ref = entity_ref
        self._users = users if users else list()
        self._project_access = project_access
        self._operation_type = operation_type
        self._created = parse_isotime(created) if created else None
        self._updated = parse_isotime(updated) if updated else None

    @property
    def acl_ref(self):
        return ACL.get_acl_ref_from_entity_ref(self.entity_ref)

    @property
    def acl_ref_relative(self):
        return self._parent_acl.acl_ref_relative

    @property
    def entity_ref(self):
        return self._entity_ref

    @property
    def entity_uuid(self):
        return self._parent_acl.entity_uuid

    @property
    def project_access(self):
        """Flag indicating project access behavior is enabled or not"""
        return self._project_access

    @property
    def users(self):
        """List of users for this ACL setting"""
        return self._users

    @property
    def operation_type(self):
        """Type indicating class of Barbican operations for this ACL"""
        return self._operation_type

    @property
    def created(self):
        return self._created

    @property
    def updated(self):
        return self._updated

    @operation_type.setter
    def operation_type(self, value):
        self._operation_type = value

    @project_access.setter
    def project_access(self, value):
        self._project_access = value

    @users.setter
    def users(self, value):
        self._users = value

    def remove(self):
        """Remove operation specific setting defined for a secret or container

        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        """
        LOG.debug('Removing {0} operation specific ACL for href: {1}'.format(self.operation_type, self.acl_ref))
        self._parent_acl.load_acls_data()
        acl_entity = self._parent_acl
        per_op_acl = acl_entity.get(self.operation_type)
        if per_op_acl:
            acl_entity.operation_acls.remove(per_op_acl)
            if acl_entity.operation_acls:
                acl_entity.submit()
            else:
                acl_entity.remove()

    def _validate_users_type(self):
        if self.users and (not (type(self.users) is list or type(self.users) is set)):
            raise ValueError('Users value is expected to be provided as list/set.')