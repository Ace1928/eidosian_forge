import sys
from unittest import mock
from libcloud.test import unittest
from libcloud.compute.drivers.libvirt_driver import LibvirtNodeDriver, have_libvirt
def _assert_arp_table(self, arp_table):
    self.assertIn('52:54:00:bc:f9:6c', arp_table)
    self.assertIn('52:54:00:04:89:51', arp_table)
    self.assertIn('52:54:00:c6:40:ec', arp_table)
    self.assertIn('52:54:00:77:1c:83', arp_table)
    self.assertIn('1.2.10.80', arp_table['52:54:00:bc:f9:6c'])
    self.assertIn('1.2.10.33', arp_table['52:54:00:04:89:51'])
    self.assertIn('1.2.10.97', arp_table['52:54:00:c6:40:ec'])
    self.assertIn('1.2.10.40', arp_table['52:54:00:77:1c:83'])