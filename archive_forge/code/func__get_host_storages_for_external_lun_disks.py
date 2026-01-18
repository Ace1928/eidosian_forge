from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _get_host_storages_for_external_lun_disks(connection):
    host_storages = {}
    hosts_service = connection.system_service().hosts_service()
    hosts_list = hosts_service.list(search='status=up')
    for host in hosts_list:
        host_storages_service = hosts_service.host_service(host.id).storage_service().list()
        for host_storage in host_storages_service:
            if host_storage.id not in host_storages.keys():
                host_storages[host_storage.id] = host_storage
    return host_storages