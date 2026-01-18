from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
import ssl
import traceback
def _ros_api_connect(module, username, password, host, port, use_tls, force_no_cert, validate_certs, validate_cert_hostname, ca_path, encoding, timeout):
    """Connect to RouterOS API."""
    if not port:
        if use_tls:
            port = 8729
        else:
            port = 8728
    try:
        params = dict(username=username, password=password, host=host, port=port, encoding=encoding, timeout=timeout)
        if use_tls:
            ctx = ssl.create_default_context(cafile=ca_path)
            wrap_context = ctx.wrap_socket
            if force_no_cert:
                ctx.check_hostname = False
                ctx.set_ciphers('ADH:@SECLEVEL=0')
            elif not validate_certs:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            elif not validate_cert_hostname:
                ctx.check_hostname = False
            else:

                def wrap_context(*args, **kwargs):
                    kwargs.pop('server_hostname', None)
                    return ctx.wrap_socket(*args, server_hostname=host, **kwargs)
            params['ssl_wrapper'] = wrap_context
        api = connect(**params)
    except Exception as e:
        connection = {'username': username, 'hostname': host, 'port': port, 'ssl': use_tls, 'status': 'Error while connecting: %s' % to_native(e)}
        module.fail_json(msg=connection['status'], connection=connection)
    return api