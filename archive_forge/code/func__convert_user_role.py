from paste.util import ip4
def _convert_user_role(self, username, roles):
    if roles and isinstance(roles, str):
        roles = roles.split(',')
    return (username, roles)