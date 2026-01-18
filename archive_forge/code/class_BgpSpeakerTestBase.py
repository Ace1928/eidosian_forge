import logging
import sys
import unittest
from os_ken.tests.integrated.common import docker_base as ctn_base
from os_ken.tests.integrated.common import oskenbgp
from os_ken.tests.integrated.common import quagga
class BgpSpeakerTestBase(unittest.TestCase):
    images = []
    containers = []
    bridges = []
    checktime = 120

    def setUp(self):
        self.skipTest('These tests require to have "docker" configured in the system. Regardless of if we move them to functional or fix them, now we need to disable them.')
        self.brdc1 = ctn_base.Bridge(name='brip6dc1', subnet='2001:10::/32')
        self.bridges.append(self.brdc1)
        self.dockerimg = ctn_base.DockerImage()
        image = 'python:%d.%d' % (sys.version_info.major, sys.version_info.minor)
        self.r_img = self.dockerimg.create_os_ken(image=image, check_exist=True)
        self.images.append(self.r_img)
        self.q_img = 'osrg/quagga'
        self.images.append(self.q_img)
        self.r1 = oskenbgp.OSKenBGPContainer(name='r1', asn=64512, router_id='192.168.0.1', ctn_image_name=self.r_img)
        self.containers.append(self.r1)
        self.r1.add_route('fc00:10::/64', route_info={'rf': 'ipv6'})
        self.r1.run(wait=True)
        self.r1_ip_cidr = self.brdc1.addif(self.r1)
        self.r1_ip = self.r1_ip_cidr.split('/')[0]
        self.q1 = quagga.QuaggaBGPContainer(name='q1', asn=64522, router_id='192.168.0.2', ctn_image_name=self.q_img)
        self.containers.append(self.q1)
        self.q1.add_route('fc00:100::/64', route_info={'rf': 'ipv6'})
        self.q1.run(wait=True)
        self.q1_ip_cidr = self.brdc1.addif(self.q1)
        self.q1_ip = self.q1_ip_cidr.split('/')[0]
        self.r1.add_peer(self.q1, bridge=self.brdc1.name, v6=True)
        self.q1.add_peer(self.r1, bridge=self.brdc1.name, v6=True)
        super().setUp()

    @classmethod
    def tearDownClass(cls):
        for ctn in cls.containers:
            try:
                ctn.stop()
            except ctn_base.CommandError as e:
                LOG.exception('Exception when stopping containers: %s', e)
            ctn.remove()
        for br in cls.bridges:
            br.delete()
        super(BgpSpeakerTestBase, cls).tearDownClass()