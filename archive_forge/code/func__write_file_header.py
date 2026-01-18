from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _write_file_header(f, url, username, ca):
    """
    Add header for paramter file, for example:
       dr_sites_primary_url: "http://engine1.redhat.com:8080/ovirt-engine/api"
       dr_sites_primary_username: "admin@internal"
       dr_sites_primary_ca_file: "ovirt-share/etc/pki/ovirt-engine/ca.pem"

       dr_sites_secondary_url:
       dr_sites_secondary_username:
       dr_sites_secondary_ca_file:
     """
    f.write('---\n')
    f.write('dr_sites_primary_url: %s\n' % url)
    f.write('dr_sites_primary_username: %s\n' % username)
    f.write('dr_sites_primary_ca_file: %s\n\n' % ca)
    f.write('# Please fill in the following properties for the secondary site: \n')
    f.write('dr_sites_secondary_url: # %s\n' % url)
    f.write('dr_sites_secondary_username: # %s\n' % username)
    f.write('dr_sites_secondary_ca_file: # %s\n\n' % ca)