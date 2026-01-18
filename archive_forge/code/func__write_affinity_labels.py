from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _write_affinity_labels(f, affinity_labels):
    f.write('\n# Mapping for affinity label\n')
    f.write('dr_affinity_label_mappings:\n')
    for affinity_label in affinity_labels:
        f.write('- primary_name: %s\n' % affinity_label)
        f.write("  # Fill the correlated affinity label name in the secondary site for affinity label '%s'\n" % affinity_label)
        f.write('  secondary_name: # %s\n\n' % affinity_label)