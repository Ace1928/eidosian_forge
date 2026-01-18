from cliff import command
from cliff import lister
from barbicanclient.v1 import acls
class AddACLUsers(lister.Lister, ArgMixin):
    """Add ACL users to a secret or container as identified by its href."""

    def get_parser(self, prog_name):
        parser = super(AddACLUsers, self).get_parser(prog_name)
        self.add_ref_arg(parser)
        self.add_per_acl_args(parser)
        return parser

    def take_action(self, args):
        """Add users to a secret or a container ACL defined in Barbican

        Provided users are added to existing ACL users if there. If input users
        is None or empty list, no change is made in existing ACL users list.
        If input project_access flag is None, then no change is made in
        existing project access behavior.

        :returns: List of objects for valid entity_ref
        :rtype: :class:`barbicanclient.acls.SecretACL` or
            :class:`barbicanclient.acls.ContainerACL`
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
        input_entity = self.create_acl_entity_from_args(self.app.client_manager.key_manager.acls, args)
        server_entity = self.app.client_manager.key_manager.acls.get(input_entity.entity_ref)
        for input_acl in input_entity.operation_acls:
            server_acl = server_entity.get(input_acl.operation_type)
            if server_acl:
                if input_acl.project_access is not None:
                    server_acl.project_access = input_acl.project_access
                if input_acl.users is not None:
                    server_acl.users.extend(input_acl.users)
            else:
                server_entity.add_operation_acl(users=input_acl.users, project_access=input_acl.project_access, operation_type=input_acl.operation_type)
        server_entity.submit()
        server_entity.load_acls_data()
        return self.get_acls_as_lister(server_entity)