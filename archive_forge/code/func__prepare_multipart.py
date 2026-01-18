from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
@staticmethod
def _prepare_multipart(fields):
    """Prepares a multipart body based on a set of fields provided.

        Ideally it would have been good to use the existing 'prepare_multipart'
        found in ansible.module_utils.urls, but it takes files and encodes them
        as Base64 strings, which is not expected by Redfish services.  It also
        adds escaping of certain bytes in the payload, such as inserting '\r'
        any time it finds a standalone '
', which corrupts the image payload
        send to the service.  This implementation is simplified to Redfish's
        usage and doesn't necessarily represent an exhaustive method of
        building multipart requests.
        """

    def write_buffer(body, line):
        if isinstance(line, text_type):
            body.append(to_bytes(line, encoding='utf-8'))
        elif isinstance(line, dict):
            body.append(to_bytes(json.dumps(line), encoding='utf-8'))
        else:
            body.append(line)
        return
    boundary = ''.join((random.choice(string.digits + string.ascii_letters) for i in range(30)))
    body = []
    for form in fields:
        write_buffer(body, '--' + boundary)
        if 'filename' in fields[form]:
            name = os.path.basename(fields[form]['filename']).replace('"', '\\"')
            write_buffer(body, u'Content-Disposition: form-data; name="%s"; filename="%s"' % (to_text(form), to_text(name)))
        else:
            write_buffer(body, 'Content-Disposition: form-data; name="%s"' % form)
        write_buffer(body, 'Content-Type: %s' % fields[form]['mime_type'])
        write_buffer(body, '')
        if 'content' not in fields[form]:
            with open(to_bytes(fields[form]['filename'], errors='surrogate_or_strict'), 'rb') as f:
                fields[form]['content'] = f.read()
        write_buffer(body, fields[form]['content'])
    write_buffer(body, '--' + boundary + '--')
    write_buffer(body, '')
    return (b'\r\n'.join(body), 'multipart/form-data; boundary=' + boundary)