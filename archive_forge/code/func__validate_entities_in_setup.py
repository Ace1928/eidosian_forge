from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _validate_entities_in_setup(self, conn, setup, python_vars):
    dcs_service = conn.system_service().data_centers_service()
    dcs_list = dcs_service.list()
    clusters = []
    affinity_groups = set()
    for dc in dcs_list:
        dc_service = dcs_service.data_center_service(dc.id)
        clusters_service = dc_service.clusters_service()
        attached_clusters_list = clusters_service.list()
        for cluster in attached_clusters_list:
            clusters.append(cluster.name)
            cluster_service = clusters_service.cluster_service(cluster.id)
            affinity_groups.update(self._fetch_affinity_groups(cluster_service))
    aff_labels = self._get_affinity_labels(conn)
    aaa_domains = self._get_aaa_domains(conn)
    networks = self._get_vnic_profile_mapping(conn)
    isValid = self._validate_networks(python_vars, networks, setup)
    isValid = self._validate_entity_exists(clusters, python_vars, self.cluster_map, setup) and isValid
    isValid = self._validate_entity_exists(list(affinity_groups), python_vars, self.aff_group_map, setup) and isValid
    isValid = self._validate_entity_exists(aff_labels, python_vars, self.aff_label_map, setup) and isValid
    return isValid