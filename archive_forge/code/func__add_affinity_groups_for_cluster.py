from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _add_affinity_groups_for_cluster(cluster_service, affinity_groups):
    affinity_groups_service = cluster_service.affinity_groups_service()
    for affinity_group in affinity_groups_service.list():
        affinity_groups.append(affinity_group.name)