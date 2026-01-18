from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _get_vnic_profile_mapping(connection):
    networks = []
    vnic_profiles_service = connection.system_service().vnic_profiles_service()
    vnic_profile_list = vnic_profiles_service.list()
    for vnic_profile_item in vnic_profile_list:
        mapped_network = {}
        networks_list = connection.system_service().networks_service().list()
        network_name = ''
        for network_item in networks_list:
            if network_item.id == vnic_profile_item.network.id:
                network_name = network_item.name
                dc_name = connection.system_service().data_centers_service().data_center_service(network_item.data_center.id).get()._name
                break
        mapped_network['network_name'] = network_name
        mapped_network['network_dc'] = dc_name
        mapped_network['profile_name'] = vnic_profile_item.name
        mapped_network['profile_id'] = vnic_profile_item.id
        networks.append(mapped_network)
    return networks