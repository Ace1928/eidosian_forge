from novaclient import base
def _build_update_body(self, version, url, md5hash):
    return {'para': {'version': version, 'url': url, 'md5hash': md5hash}}