import logging
import os
import netaddr
from . import docker_base as base
class RawQuaggaBGPContainer(QuaggaBGPContainer):

    def __init__(self, name, config, ctn_image_name, zebra=False):
        asn = None
        router_id = None
        for line in config.split('\n'):
            line = line.strip()
            if line.startswith('router bgp'):
                asn = int(line[len('router bgp'):].strip())
            if line.startswith('bgp router-id'):
                router_id = line[len('bgp router-id'):].strip()
        if not asn:
            raise Exception('asn not in quagga config')
        if not router_id:
            raise Exception('router-id not in quagga config')
        self.config = config
        super(RawQuaggaBGPContainer, self).__init__(name, asn, router_id, ctn_image_name, zebra)

    def create_config(self):
        with open(os.path.join(self.config_dir, 'bgpd.conf'), 'w') as f:
            LOG.info("[%s's new config]", self.name)
            LOG.info(self.config)
            f.writelines(self.config)