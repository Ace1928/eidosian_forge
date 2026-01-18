from os import path
import sys
from oslo_log import log as logging
from oslo_serialization import jsonutils
from saharaclient.osc.v1 import job_types as jt_v1
class GetJobTypeConfigs(jt_v1.GetJobTypeConfigs):
    """Get job type configs"""
    log = logging.getLogger(__name__ + '.GetJobTypeConfigs')

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        if not parsed_args.file:
            parsed_args.file = parsed_args.job_type
        data = client.job_templates.get_configs(parsed_args.job_type).to_dict()
        if path.exists(parsed_args.file):
            self.log.error('File "%s" already exists. Choose another one with --file argument.' % parsed_args.file)
        else:
            with open(parsed_args.file, 'w') as f:
                jsonutils.dump(data, f, indent=4)
            sys.stdout.write('"%(type)s" job configs were saved in "%(file)s"file' % {'type': parsed_args.job_type, 'file': parsed_args.file})