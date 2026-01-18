import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def _get_container_info(self, instance, log_name):
    try:
        log_info = self.log_show(instance, log_name)
        container = log_info.container
        prefix = log_info.prefix
        metadata_file = log_info.metafile
        return (container, prefix, metadata_file)
    except swift_client.ClientException as ex:
        if ex.http_status == 404:
            raise exceptions.GuestLogNotFoundError()
        raise