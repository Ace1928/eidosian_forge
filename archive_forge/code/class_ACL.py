import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
class ACL(object):
    _resource_name = 'acl'

    def __init__(self, api, entity_ref, users=None, project_access=None, operation_type=DEFAULT_OPERATION_TYPE, created=None, updated=None):
        """Base ACL entity instance for secret or container.

        Provide ACL data arguments to set ACL setting for given operation_type.

        To add ACL setting for other operation types, use `add_operation_acl`
        method.

        :param api: client instance reference
        :param str entity_ref: Full HATEOAS reference to a secret or container
        :param users: List of Keystone userid(s) to be used for ACL.
        :type users: str List or None
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
        self._api = api
        self._entity_ref = entity_ref
        self._operation_acls = []
        if users is not None or project_access is not None:
            acl = _PerOperationACL(parent_acl=self, entity_ref=entity_ref, users=users, project_access=project_access, operation_type=operation_type, created=created, updated=updated)
            self._operation_acls.append(acl)

    @property
    def entity_ref(self):
        """Entity URI reference."""
        return self._entity_ref

    @property
    def entity_uuid(self):
        """Entity UUID"""
        return str(base.validate_ref_and_return_uuid(self._entity_ref, self._acl_type))

    @property
    def operation_acls(self):
        """List of operation specific ACL settings."""
        return self._operation_acls

    @property
    def acl_ref(self):
        return ACL.get_acl_ref_from_entity_ref(self.entity_ref)

    @property
    def acl_ref_relative(self):
        return ACL.get_acl_ref_from_entity_ref_relative(self.entity_uuid, self._parent_entity_path)

    def add_operation_acl(self, users=None, project_access=None, operation_type=None, created=None, updated=None):
        """Add ACL settings to entity for specific operation type.

        If matching operation_type ACL already exists, then it replaces it with
        new PerOperationACL object using provided inputs. Otherwise it appends
        new PerOperationACL object to existing per operation ACL list.

        This just adds to local entity and have not yet applied these changes
        to server.

        :param users: List of Keystone userid(s) to be used in ACL.
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
        new_acl = _PerOperationACL(parent_acl=self, entity_ref=self.entity_ref, users=users, project_access=project_access, operation_type=operation_type, created=created, updated=updated)
        for i, acl in enumerate(self._operation_acls):
            if acl.operation_type == operation_type:
                self._operation_acls[i] = new_acl
                break
        else:
            self._operation_acls.append(new_acl)

    def _get_operation_acl(self, operation_type):
        return next((acl for acl in self._operation_acls if acl.operation_type == operation_type), None)

    def get(self, operation_type):
        """Get operation specific ACL instance.

        :param str operation_type: Type indicating which operation's ACL
            setting is needed.
        """
        return self._get_operation_acl(operation_type)

    def __getattr__(self, name):
        if name in VALID_ACL_OPERATIONS:
            return self._get_operation_acl(name)
        else:
            raise AttributeError(name)

    def submit(self):
        """Submits ACLs for a secret or a container defined in server

        In existing ACL case, this overwrites the existing ACL setting with
        provided inputs. If input users are None or empty list, this will
        remove existing ACL users if there. If input project_access flag is
        None, then default project access behavior is enabled.

        :returns: str acl_ref: Full HATEOAS reference to a secret or container
            ACL.
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        LOG.debug('Submitting complete {0} ACL for href: {1}'.format(self.acl_type, self.entity_ref))
        if not self.operation_acls:
            raise ValueError('ACL data for {0} is not provided.'.format(self._acl_type))
        self.validate_input_ref()
        acl_dict = {}
        for per_op_acl in self.operation_acls:
            per_op_acl._validate_users_type()
            op_type = per_op_acl.operation_type
            acl_data = {}
            if per_op_acl.project_access is not None:
                acl_data['project-access'] = per_op_acl.project_access
            if per_op_acl.users is not None:
                acl_data['users'] = per_op_acl.users
            acl_dict[op_type] = acl_data
        response = self._api.put(self.acl_ref_relative, json=acl_dict)
        return response.json().get('acl_ref')

    def remove(self):
        """Remove Barbican ACLs setting defined for a secret or container

        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        """
        self.validate_input_ref()
        LOG.debug('Removing ACL for {0} for href: {1}'.format(self.acl_type, self.entity_ref))
        self._api.delete(self.acl_ref_relative)

    def load_acls_data(self):
        """Loads ACL entity from Barbican server using its acl_ref

        Clears the existing list of per operation ACL settings if there.
        Populates current ACL entity with ACL settings received from Barbican
        server.

        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        response = self._api.get(self.acl_ref_relative)
        del self.operation_acls[:]
        for op_type in response:
            acl_dict = response.get(op_type)
            proj_access = acl_dict.get('project-access')
            users = acl_dict.get('users')
            created = acl_dict.get('created')
            updated = acl_dict.get('updated')
            self.add_operation_acl(operation_type=op_type, project_access=proj_access, users=users, created=created, updated=updated)

    def validate_input_ref(self):
        res_title = self._acl_type.title()
        if not self.entity_ref:
            raise ValueError('{0} href is required.'.format(res_title))
        if self._parent_entity_path in self.entity_ref:
            if '/acl' in self.entity_ref:
                raise ValueError('{0} ACL URI provided. Expecting {0} URI.'.format(res_title))
            ref_type = self._acl_type
        else:
            raise ValueError('{0} URI is not specified.'.format(res_title))
        base.validate_ref_and_return_uuid(self.entity_ref, ref_type)
        return ref_type

    @staticmethod
    def get_acl_ref_from_entity_ref(entity_ref):
        if entity_ref:
            entity_ref = entity_ref.rstrip('/')
            return '{0}/{1}'.format(entity_ref, ACL._resource_name)

    @staticmethod
    def get_acl_ref_from_entity_ref_relative(entity_ref, entity_type):
        if entity_ref:
            entity_ref = entity_ref.rstrip('/')
            return '{0}/{1}/{2}'.format(entity_type, entity_ref, ACL._resource_name)

    @staticmethod
    def identify_ref_type(entity_ref):
        if not entity_ref:
            raise ValueError('Secret or container href is required.')
        if '/secrets' in entity_ref:
            ref_type = 'secret'
        elif '/containers' in entity_ref:
            ref_type = 'container'
        else:
            raise ValueError('Secret or container URI is not specified.')
        return ref_type