from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
import os
from apitools.gen import gen_client
from googlecloudsdk.api_lib.regen import api_def
from googlecloudsdk.api_lib.regen import resource_generator
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
import six
def GenerateApitoolsApi(base_dir, root_dir, api_name, api_version, api_config):
    """Invokes apitools generator for given api."""
    discovery_doc = api_config['discovery_doc']
    args = [gen_client.__file__]
    unelidable_request_methods = api_config.get('unelidable_request_methods')
    if unelidable_request_methods:
        args.append('--unelidable_request_methods={0}'.format(','.join(api_config['unelidable_request_methods'])))
    args.extend(['--init-file=empty', '--nogenerate_cli', '--infile={0}'.format(os.path.join(base_dir, root_dir, discovery_doc)), '--outdir={0}'.format(os.path.join(base_dir, root_dir, api_name, api_version)), '--overwrite', '--apitools_version=CloudSDK', '--user_agent=google-cloud-sdk', '--root_package', '{0}.{1}.{2}'.format(root_dir.replace('/', '.'), api_name, api_version), 'client'])
    logging.debug('Apitools gen %s', args)
    gen_client.main(args)
    package_dir = base_dir
    for subdir in [root_dir, api_name, api_version]:
        package_dir = os.path.join(package_dir, subdir)
        init_file = os.path.join(package_dir, '__init__.py')
        if not os.path.isfile(init_file):
            logging.warning('%s does not have __init__.py file, generating ...', package_dir)
            files.WriteFileContents(init_file, _INIT_FILE_CONTENT)