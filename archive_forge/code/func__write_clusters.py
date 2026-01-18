from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _write_clusters(f, clusters):
    f.write('# Mapping for cluster\n')
    f.write('dr_cluster_mappings:\n')
    for cluster_name in clusters:
        f.write('- primary_name: %s\n' % cluster_name)
        f.write("  # Fill the correlated cluster name in the secondary site for cluster '%s'\n" % cluster_name)
        f.write('  secondary_name: # %s\n\n' % cluster_name)