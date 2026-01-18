import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def _log_generator(instance, log_name, lines, swift):
    try:
        container, prefix, metadata_file = self._get_container_info(instance, log_name)
        head, body = swift.get_container(container, prefix=prefix)
        log_obj_to_display = []
        if lines:
            total_lines = lines
            partial_results = False
            parts = sorted(body, key=lambda obj: obj['last_modified'], reverse=True)
            for part in parts:
                obj_hdrs = swift.head_object(container, part['name'])
                obj_lines = int(obj_hdrs['x-object-meta-lines'])
                log_obj_to_display.insert(0, part)
                if obj_lines >= lines:
                    partial_results = True
                    break
                lines -= obj_lines
            if not partial_results:
                lines = total_lines
            part = log_obj_to_display.pop(0)
            hdrs, log_obj = swift.get_object(container, part['name'])
            log_by_lines = log_obj.decode().splitlines()
            yield ('\n'.join(log_by_lines[-1 * lines:]) + '\n')
        else:
            log_obj_to_display = sorted(body, key=lambda obj: obj['last_modified'])
        for log_part in log_obj_to_display:
            headers, log_obj = swift.get_object(container, log_part['name'])
            yield log_obj.decode()
    except swift_client.ClientException as ex:
        if ex.http_status == 404:
            raise exceptions.GuestLogNotFoundError()
        raise