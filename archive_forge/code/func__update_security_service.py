from manilaclient import api_versions
from manilaclient import base
from manilaclient import exceptions
def _update_security_service(self, security_service, dns_ip=None, ou=None, server=None, domain=None, password=None, user=None, name=None, description=None, default_ad_site=None):
    values = {}
    if dns_ip is not None:
        values['dns_ip'] = dns_ip
    if ou is not None:
        values['ou'] = ou
    if server is not None:
        values['server'] = server
    if domain is not None:
        values['domain'] = domain
    if user is not None:
        values['user'] = user
    if password is not None:
        values['password'] = password
    if name is not None:
        values['name'] = name
    if description is not None:
        values['description'] = description
    if default_ad_site is not None:
        values['default_ad_site'] = default_ad_site
    for k, v in values.items():
        if v == '':
            values[k] = None
    if not values:
        msg = 'Must specify fields to be updated'
        raise exceptions.CommandError(msg)
    body = {RESOURCE_NAME: values}
    return self._update(RESOURCE_PATH % base.getid(security_service), body, RESOURCE_NAME)