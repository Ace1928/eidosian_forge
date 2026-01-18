import logging
import os
import stat
from ray.autoscaler._private.aliyun.utils import AcsClient
def _get_or_import_key_pair(config):
    cli = _client(config)
    key_name = config['provider'].get('key_name', 'ray')
    key_path = os.path.expanduser('~/.ssh/{}'.format(key_name))
    keypairs = cli.describe_key_pairs(key_pair_name=key_name)
    if keypairs is not None and len(keypairs) > 0:
        if 'ssh_private_key' not in config['auth']:
            logger.info('{} keypair exists, use {} as local ssh key'.format(key_name, key_path))
            config['auth']['ssh_private_key'] = key_path
    elif 'ssh_private_key' not in config['auth']:
        resp = cli.create_key_pair(key_pair_name=key_name)
        if resp is not None:
            with open(key_path, 'w+') as f:
                f.write(resp.get('PrivateKeyBody'))
            os.chmod(key_path, stat.S_IRUSR)
            config['auth']['ssh_private_key'] = key_path
    else:
        public_key_file = config['auth']['ssh_private_key'] + '.pub'
        with open(public_key_file) as f:
            public_key = f.readline().strip('\n')
            cli.import_key_pair(key_pair_name=key_name, public_key_body=public_key)
            return