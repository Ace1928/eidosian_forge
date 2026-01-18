from __future__ import absolute_import, division, print_function
import abc
from ansible.module_utils import six
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
class CertificateChain(object):
    """
    Download and parse the certificate chain.
    https://tools.ietf.org/html/rfc8555#section-7.4.2
    """

    def __init__(self, url):
        self.url = url
        self.cert = None
        self.chain = []
        self.alternates = []

    @classmethod
    def download(cls, client, url):
        content, info = client.get_request(url, parse_json_result=False, headers={'Accept': 'application/pem-certificate-chain'})
        if not content or not info['content-type'].startswith('application/pem-certificate-chain'):
            raise ModuleFailException('Cannot download certificate chain from {0}, as content type is not application/pem-certificate-chain: {1} (headers: {2})'.format(url, content, info))
        result = cls(url)
        certs = split_pem_list(content.decode('utf-8'), keep_inbetween=True)
        if certs:
            result.cert = certs[0]
            result.chain = certs[1:]
        process_links(info, lambda link, relation: result._process_links(client, link, relation))
        if result.cert is None:
            raise ModuleFailException('Failed to parse certificate chain download from {0}: {1} (headers: {2})'.format(url, content, info))
        return result

    def _process_links(self, client, link, relation):
        if relation == 'up':
            if not self.chain:
                chain_result, chain_info = client.get_request(link, parse_json_result=False)
                if chain_info['status'] in [200, 201]:
                    self.chain.append(der_to_pem(chain_result))
        elif relation == 'alternate':
            self.alternates.append(link)

    def to_json(self):
        cert = self.cert.encode('utf8')
        chain = '\n'.join(self.chain).encode('utf8')
        return {'cert': cert, 'chain': chain, 'full_chain': cert + chain}